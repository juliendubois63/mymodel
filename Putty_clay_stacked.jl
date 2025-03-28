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


# Define the function
function steady_state_solver(m_ss::Number, params::NamedTuple)
    α, β, δ, ϵ, σ, g_A, g_Ae, g, R₀ = params.α, params.β, params.δ, params.ϵ, params.σ, params.g_A, params.g_Ae, params.g, params.R₀

    function equations(vars)
        v, x, λ = vars
        r1 = (1-g) * R₀ - x * (f(v, ϵ) / v) * (1 / δ)
        r2 = ((f(v, ϵ) - f_prime(v, ϵ) * v) / (1 - β * (1 - δ))) * (((1/δ)*(f(v,ϵ)) - 1)*x)^(-σ) - λ * (x / (1 - g * (1 - δ)))
        r3 = (((1/δ)*(f(v,ϵ)) - 1)*x)^(-σ) + (λ / (1 - g * (1 - δ))) - (f(v, ϵ) / v) * (((1/δ)*(f(v,ϵ)) - 1)*x)^(-σ) * (1 / (1 - β * (1 - δ)))
        return [r1, r2, r3]
    end

    initial_guess = [0.5,0.5,0.5]
    solution = nlsolve(equations, initial_guess)
    return solution.zero
end

m_ss = (1-g)*R₀
# Solve for steady state
steady_state_values = steady_state_solver(m_ss, params)
v_ss, x_ss, λ_ss = steady_state_values

v̄_ss = ((λ_ss/(((((1-δ)/δ)*(v_ss^((ϵ - 1)/ϵ) + 1)^(ϵ/(ϵ-1)) - 1)*x_ss))^-(params.σ))^((params.ϵ - 1)/params.ϵ) - 1)^(params.ϵ/(params.ϵ-1))

(1-g) * R₀ - x_ss * (f(v_ss, ϵ) / v_ss) * (1 / δ)
((f(v_ss, ϵ) - f_prime(v_ss, ϵ) * v_ss) / (1 - β * (1 - δ))) * (((1/δ)*(f(v_ss,ϵ)) - 1)*x_ss)^(-σ) - λ * (x_ss / (1 - g * (1 - δ)))
(((1/δ)*(f(v_ss,ϵ)) - 1)*x_ss)^(-σ) + (λ_ss / (1 - g * (1 - δ))) - (f(v_ss, ϵ) / v_ss) * (((1/δ)*(f(v_ss,ϵ)) - 1)*x_ss)^(-σ) * (1 / (1 - β * (1 - δ)))

#Initial guess 
T = 100
x_0, v_0 = 1.0, 1.0
v_0 = [x_0, v_0]
v_T = [v_ss, x_ss]

# Create v initial guess as a T×3 matrix  
v0 = vcat(fill(v_T', T)...)  # Create a T×3 matrix
v0 = Matrix{Number}(v0)     # Ensure v0 is a Matrix{Number}
λ = λ_ss                    # Initial guess for λ



# function state(t::Int, v::Vector, x::Vector, λ::Number, params)
#     sum_term = t > 1 ? sum((1 - δ)^(t - k) * x[k] * f(v[k], params.ϵ) for k in 1:t-1) : 0.0
#     v̄[t] = (((λ / ((1 - δ) * sum_term - x[t])^(-params.σ)) * (params.g / params.β)^t)^((params.ϵ - 1) / params.ϵ) - 1)^(params.ϵ / (params.ϵ - 1))
#     return v̄[t]
# end
# state(3, Vector{Number}(v0[:, 1]), Vector{Number}(v0[:, 2]), λ, params)

v̄ = zeros(T)  # Initialize v̄ as a vector of zeros with length T
# for i in 1:T
#     println(state(i, Vector{Number}(v0[:, 1]), Vector{Number}(v0[:, 2]), v̄, λ, params))
# end

t = 100

SU_t1 = sum((β*(1-δ))^s * (((1-δ)*sum((1-δ)^(t+1-s-k) * v0[k, 2] * (f(v0[k, 1], ϵ) / v0[k, 1]) * (k <= T && v0[k, 1] >= v̄[min(t+s, T)] ? 1 : 0) for k in 1:min(t+s, T)) - (t+s <= T ? v0[t+s, 2] : 0))^(-σ)) for s in 1:(T-1); init=0.0)

SU_t = sum((β*(1-δ))^s * (((1-δ)*sum((1-δ)^(t+s-k) * v0[k, 2] * (f(v0[k, 1], ϵ) / v0[k, 1]) * (k <= T && v0[k, 1] >= v̄[min(t+s-1, T)] ? 1 : 0) for k in 1:min(t+s-1, T)) - (t+s-1 <= T ? v0[t+s-1, 2] : 0))^(-σ)) for s in 1:(T-1); init=0.0)

SC_t1 = sum((β * (1 - δ) * g)^i for i in 1:T if i <= T && v0[i, 1] >= v0[min(t + 1 + i, T), 1] && t + 1 + i <= T; init=0.0)

SC_t = sum((β * (1 - δ) * g)^i for i in 1:T if i <= T && v0[i, 1] >= v0[min(t + i, T), 1] && t + i <= T; init=0.0)

c_t1 = ((1-δ)*sum((1-δ)^(t+1-k) * v0[k, 2] * (f(v0[k, 1], ϵ) / v0[k, 1]) for k in 1:min(t+1, T)) - (t+1 <= T ? v0[t+1, 2] : 0))

c_t = ((1-δ)*sum((1-δ)^(t-k) * v0[k, 2] * (f(v0[k, 1], ϵ) / v0[k, 1]) for k in 1:min(t, T); init=0.0) - (t <= T ? v0[t, 2] : 0))

c_t = 0.0
for k in 1:T
    term = (1-δ)^(t-k) * v0[k, 2] * (f(v0[k, 1], ϵ) / v0[k, 1])
    println("k = $k, term = $term")
    if k <= t
        c_t += term
    end
end
c_t -= (t <= T ? v0[t, 2] : 0)
println("Final c_t = $c_t")


function Transition(t::Int, v::Vector, x::Vector, v̄::Vector, λ::Number, params)

    SU_t1 = sum((β*(1-δ))^s * (((1-δ)*sum((1-δ)^(t+1-s-k) * x[k] * (f(v[k], ϵ) / v[k]) * (k <= T && v[k] >= v̄[min(t+s, T)] ? 1 : 0) for k in 1:min(t+s, T)) - (t+s <= T ? x[t+s] : 0))^(-σ)) for s in 1:(T-1); init=0.0)

    SU_t = sum((β*(1-δ))^s * (((1-δ)*sum((1-δ)^(t+s-k) * x[k] * (f(v[k], ϵ) / v[k]) * (k <= T && v[k] >= v̄[min(t+s-1, T)] ? 1 : 0) for k in 1:min(t+s-1, T)) - (t+s-1 <= T ? x[t+s-1] : 0))^(-σ)) for s in 1:(T-1); init=0.0)

    SC_t1 = sum((β * (1 - δ) * g)^i for i in 1:T if i <= T && v[i] >= v[min(t + 1 + i, T)] && t + 1 + i <= T; init=0.0)

    SC_t = sum((β * (1 - δ) * g)^i for i in 1:T if i <= T && v[i] >= v[min(t + i, T)] && t + i <= T; init=0.0)

    c_t1 = ((1-δ)*sum((1-δ)^(t+1-k) * x[k] * (f(v[k], ϵ) / v[k]) * (k <= T && v[k] >= v̄[min(t+1+k, T)] ? 1 : 0) for k in 1:min(t+1, T); init=0.0) - (t+1 <= T ? x[t+1] : 0))

    c_t = ((1-δ)*sum((1-δ)^(t-k) * x[k] * (f(v[k], ϵ) / v[k]) * (k <= T && v[k] >= v̄[min(t+k, T)] ? 1 : 0) for k in 1:min(t, T); init=0.0) - (t <= T ? x[t] : 0))

    res1 = (f(v[min(t+1, T)], params.ϵ) - f_prime(v[min(t+1, T)], params.ϵ) * v[min(t+1, T)]) /
           (f(v[min(t, T)], params.ϵ) - f_prime(v[min(t, T)], params.ϵ) * v[min(t, T)]) *
           (SU_t1 / SU_t) - params.g / params.β *
           (t+1 <= T ? x[t+1] : 0) / (t <= T ? x[t] : 0) * (SC_t1 / SC_t)

    res2 = (SU_t1 / SU_t) -
           ((c_t1)^(-σ) + λ*(g/β)^(t+1) * SC_t1)/((c_t)^(-σ) + λ*(g/β)^t * SC_t)

    return res1, res2
end



Transition(10, Vector{Number}(v0[:, 1]), Vector{Number}(v0[:, 2]), v̄, λ, params)

# Define the matrix of residuals
function F(v::Matrix{Number}, λ::Number)
    R = zeros(eltype(v), T, 2)  
    R[T, 1] = x_ss - v[T, 1] # Fix the last value of residuals as a difference to x_ss
    R[T, 2] = v_ss - v[T, 2] # Fix the last value of residuals as a difference to v_ss
    #R[1, 3] = v̄_0 - v[1, 3] # Fix the first value of residuals as a difference to v̄_0

    for t in 1:T-1
        # Call Transition and assign the results to temporary variables
        res1, res2 = Transition(t, Vector{Number}(v[:, 1]), Vector{Number}(v[:, 2]), v̄, λ, params)
        R[t, 1] = res1
        R[t, 2] = res2

        # Call state and assign the result to R[t+1, 3]
        #R[t+1, 3] = state(t, Vector{Number}(v[:, 1]), Vector{Number}(v[:, 2]), Vector{Number}(v[:, 3]), λ, params)
    end

    return R
end


# Test the F function
R = F(v0, λ)
println("Residual matrix R: ", R)

ForwardDiff.jacobian(v -> F(v, λ), v0)

#Solve F(v, λ) = 0 without nlsolve using newton 
function newton(F, v0, λ, tol = 1e-6, max_iter = 1000)
    v = copy(v0)
    for iter in 1:max_iter
        R = F(v, λ)
        if norm(R) < tol
            break
        end
        # Compute the Jacobian manually
        J = zeros(eltype(v), size(v, 1) * size(v, 2), size(v, 1) * size(v, 2))
        h = 1e-8  # Small perturbation for finite differences
        for i in 1:size(v, 1)
            for j in 1:size(v, 2)
                v_perturbed = copy(v)
                v_perturbed[i, j] += h
                R_perturbed = F(v_perturbed, λ)
                J[:, (i - 1) * size(v, 2) + j] = vec((R_perturbed - R) / h)
            end
        end
        Δv = reshape(-J \ vec(R), size(v))
        v += Δv
    end
    return v
end

v_sol = newton(F, v0, λ)


# Solve using nlsolve
res = nlsolve(x -> F(x, λ), v0)


# Plot the results
function plot_results(v::Matrix)
    p1 = plot(v[:, 1], label = "k", title = "output")
    p2 = plot(v[:, 2], label = "e", title = "total energy")
    p3 = plot(v[:, 3], label = "c", title = "investment")
    p4 = plot(v[:, 4], label = "v", title = "energy intensity")
    p5 = plot(v[:, 5], label = "v̄", title = "threshold energy intensity")
    p6 = plot(v[:, 6], label = "c", title = "consumption")
    plot(p1, p2, p3, p4, p5, p6, layout = (3, 2))
end

plot_results(v_sol)