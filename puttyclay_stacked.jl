using Plots
using NLsolve
using ForwardDiff
using SparseArrays
using LinearAlgebra
using Optim

# Define parameters
α = 0.33
σ = 1.0
ϵ = 0.02
β = 0.985
δ = 0.05
R₀ = 1000
g_A = 1.01

# Adjust β and compute g_Ae
β *= g_A^((1-σ)/(1-α))
g_Ae = g_A^(σ/(1-α)) / β
g = g_A^(1/(1-α))/g_Ae

params = (; α, β, δ, ϵ, σ, g_A, g_Ae, g, R₀)

# Utility function
function u(c::Number, σ::Number)
    return isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
end

# Utility function and derivative
function u_prime(c::Number, σ::Number)
    c = ForwardDiff.derivative(x -> u(x, σ), c)
    return c
end

#Production function
function f(v::Number, ϵ::Number)
    return (v^((ϵ - 1)/ϵ)+1)^(ϵ/(ϵ-1))
end

#Production function derivative
function f_prime(v::Number, ϵ::Number)
    v = ForwardDiff.derivative(x -> f(x, ϵ), v)
    return v    
end

m_ss = (1-g) * R₀
# Define the function
# Define the function
function steady_state_solver(m_ss::Number, params::NamedTuple)
    α, β, δ, ϵ, σ, g_A, g_Ae, g, R₀ = params.α, params.β, params.δ, params.ϵ, params.σ, params.g_A, params.g_Ae, params.g, params.R₀

    function equations(vars)
        y, c, v, x, λ = vars
        r1 = m_ss - x * (1/ v) * (1 / δ)
        r2 = y - x * (f(v, ϵ) / v) * (1 / δ)
        r3 = ((f(v, ϵ) - f_prime(v, ϵ) * v) / (1 - β * (1 - δ))) * c^(-σ) - λ * (x / (1 - g*(1 - δ)))
        r4 = c^(-σ) + (λ / (1 - g*(1 - δ))) - (f(v, ϵ) / v) * c^(-σ) * (1 / (1 - β*(1 - δ)))
        r5 = y - c - x
        return [r1, r2, r3, r4, r5]
    end

    initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
    solution = nlsolve(equations, initial_guess)
    return solution.zero
end

# Solve for steady state
steady_state_values = steady_state_solver(m_ss, params)
y_ss, c_ss, v_ss, x_ss, λ_ss = steady_state_values

v̄_ss = ((λ_ss/c_ss^-(params.σ))^((params.ϵ - 1)/params.ϵ) - 1)^(params.ϵ/(params.ϵ-1))

# State Transition
function state(y, m, x, v, params::NamedTuple)
    
    y_next = (1-δ)y + x*(f(v,params.ϵ)/v)
    m_next = (1-δ)m + x*(1/v)

    return y_next, m_next
end


# state(v0[50,1], v0[50,2],v0[50,3], v0[50,4], params)



# Control Transition
function control(t::Int, x::Vector, v::Vector, c::Vector, y::Vector, params::NamedTuple)
    α, β, δ, ϵ, σ, g_A, g_Ae, g, R₀ = params.α, params.β, params.δ, params.ϵ, params.σ, params.g_A, params.g_Ae, params.g, params.R₀
    
    #define obsolescence threshold values
    v̄ = zeros(eltype(v), T)
    for i in 1:T
        try
            v̄[i] = ((λ_ss/c[i]^-(params.σ))^((params.ϵ - 1)/params.ϵ) - 1)^(params.ϵ/(params.ϵ-1))
        catch e
            if isa(e, DomainError)
                v̄[i] = 0
            else
                rethrow(e)
            end
        end
    end

    #define length of use of vintage t
    i = 0
    while i < T && v[t] > v̄[min(t+i, T)]
        i += 1
    end

    #define length of use of vintage t+1
    k = 0
    while k < T && v[t+1] > v̄[min(t+1+k, T)]
        k += 1
    end

    SU_t1 = sum((β*(1-δ))^s * c[min(t+1+s, T)]^(-σ) for s in 1:k, init = 0.0)
    SU_t = sum((β*(1-δ))^s * c[min(t+s, T)]^(-σ) for s in 1:i, init = 0.0)

    SC_t1 = sum((β*(1-δ)*g)^s for s in 1:k, init = 0.0)
    SC_t = sum((β*(1-δ)*g)^s for s in 1:i, init = 0.0)

    res1 = x[t+1] - (f(v[t+1], ϵ) - f_prime(v[t+1], ϵ) * v[t+1])/(f(v[t], ϵ) - f_prime(v[t], ϵ) * v[t]) * (SU_t1)/(SU_t)*(SC_t/SC_t1)* (β/g) * x[t] 
    res2 = c[t+1]^(-σ) + λ*(g/β)^(t+1) * SC_t1 - (SU_t1/SU_t)*(c[t]^(-σ) + λ*(g/β)^t * SC_t)
    res3 = x[t+1] - y[t+1] + c[t+1]

    return res1, res2, res3
end

# control(2, v0[:,3], v0[:,4], v0[:,5], v0[:,1], params)

# for t in 1:T-1
#     SU_t1 = sum((1/(1-β*(1-δ)))* v0[i, 1]^(-σ) for i in t+1:T, init = 0.0)
#     SU_t = sum((1/(1-β*(1-δ))) * v0[i, 1]^(-σ) for i in t:T, init = 0.0)
#     ratio = SU_t1 / SU_t
#     println("At t = $t: SU_t1 = $SU_t1, SU_t = $SU_t, Ratio (SU_t1 / SU_t) = $ratio")
# end

#loop to compute control values for a range of t
# for k in 1:T-1
#     test = control(k, Vector{Number}(v0[:, 3]), Vector{Number}(v0[:, 4]), Vector{Number}(v0[:, 5]), Vector{Number}(v0[:, 1]), params) 
#     println("at iteration $k :", test)
# end


# Define the matrix of residuals
function F(v::Matrix) 
    R = zeros(eltype(v), T, 5)
    R[1, 1] = v[1, 1] - 14.0 
    R[1, 2] = v[1, 2] - 14.0  
    R[T, 3] = v_ss - v[T, 4] 
    R[T, 4] = x_ss - v[T, 3] 
    R[T, 5] = c_ss - v[T, 5] 

    for t in 1:T-1
        R[t+1, 1] = v[t+1,1] - state(v[t, 1], v[t, 2], v[t, 3], v[t, 4], params)[1]
        R[t+1, 2] = v[t+1,2] - state(v[t, 1], v[t, 2], v[t, 3], v[t, 4], params)[2]
        R[t, 3], R[t, 4], R[t, 5] = control(t, Vector{Number}(v[:, 3]), Vector{Number}(v[:, 4]), Vector{Number}(v[:, 5]), Vector{Number}(v[:, 1]), params)
    end

    return R
end


#Initial guess 
T = 100
h = 1e-6
y_0, m_0, x_0, v_0, c_0 = y_ss + h, m_ss + h, x_ss + h, v_ss + h, c_ss + h

v_0 = [y_0, m_0, x_0, v_0, c_0]

v_T = [y_ss, m_ss, x_ss, v_ss, c_ss]


#Define v as a linear increase from v_0 to v_T
v0 = [v_0 + (v_T - v_0) * t / T for t in 0:T-1]
# Convert v to a matrix
v0 = hcat(v0...)'
v0 = Matrix(v0)


# Create steady_state initial guess as a 100x5 matrix
v0 = vcat(fill(v_T', T)...) # Initial guess for v (100x5 matrix)
#convert v_0 to a Matrix{Number}
v0 = Matrix(v0)

λ = λ_ss       # Initial guess for λ (scalar)


# Test the F function
F(v0)

# Solve using nlsolve
res = nlsolve(x -> F(x), v0,autodiff=:forward, show_trace = true,  iterations=100)
v_sol = res.zero

#manual newton solver for F(x) = 0, initial guess v0
# function newton_solver(F, v0; tol = 1e-6, max_iter = 1000, damping_factor = 0.5)
#     v = v0
#     norm_res = norm(vec(F(v)))  # Flatten residuals into a vector
#     iter = 0
#     h = 1e-8  # Small perturbation for finite differences

#     while norm_res > tol && iter < max_iter
#         println("Iteration $iter: Residual norm = $norm_res")  # Print iteration info

#         # Compute the Jacobian manually
#         F_v = vec(F(v))  # Flatten residuals into a vector
#         J = zeros(eltype(v), length(F_v), length(vec(v)))  # Jacobian for flattened residuals

#         for i in 1:size(v, 1)
#             for j in 1:size(v, 2)
#                 v_perturbed = copy(v)
#                 v_perturbed[i, j] += h
#                 F_perturbed = vec(F(v_perturbed))  # Flatten perturbed residuals
#                 J[:, (i - 1) * size(v, 2) + j] = (F_perturbed - F_v) / h
#             end
#         end

#         Δv = reshape(-J \ F_v, size(v))  # Solve for Δv and reshape to match v
#         v += damping_factor * Δv  # Apply damping factor to the update
#         norm_res = norm(vec(F(v)))  # Update residual norm
#         iter += 1
#     end

#     if norm_res <= tol
#         println("Converged in $iter iterations with residual norm = $norm_res")
#     else
#         println("Failed to converge in $max_iter iterations. Final residual norm = $norm_res")
#     end

#     return v
# end
# newton_solver(x -> F(x), v0)

# Plot the results
function plot_results(v::Matrix)
    p1 = plot(v[:, 1], label = "k", title = "output")
    p2 = plot(v[:, 2], label = "e", title = "total energy")
    p3 = plot(v[:, 3], label = "c", title = "investment")
    p4 = plot(v[:, 4], label = "v", title = "energy intensity")
    p5 = plot(v[:, 5], label = "c", title = "consumption")
    plot(p1, p2, p3, p4, p5,  layout = (3, 2))
end

plot_results(v_sol)