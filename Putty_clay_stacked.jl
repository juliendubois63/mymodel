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
        r1 = m_ss - x * (f(v, ϵ) / v) * (1 / δ)
        r2 = y - x * (f(v, ϵ) / v) * (1 / δ)
        r3 = ((f(v, ϵ) - f_prime(v, ϵ) * v) / (1 - β * (1 - δ))) * c^(-σ) - λ * (x / (1 - g * (1 - δ)))
        r4 = c^(-σ) + (λ / (1 - g * (1 - δ))) - (f(v, ϵ) / v) * c^(-σ) * (1 / (1 - β * (1 - δ)))
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
function state(t::Int, x::Vector{Number}, v::Vector{Number}, c::Vector{Number}, λ::Number, params::NamedTuple)
    v̄ = [(λ / c[i]^-(params.σ))^((params.ϵ - 1)/params.ϵ) - 1 for i in 1:length(v)]
    X = zeros(length(x), length(x))
    V = zeros(length(v), length(v))
    for i in 1:length(v)
        for j in 1:i
            X[j, i] = x[j]
            V[j, i] = v[j]
        end
    end 
    for i in 1:length(v)
        for j in 1:i
            if V[j, i] < v̄[i]
                X[j, i] = 0
            end
        end
    end

    y = sum((1-params.δ)^i * X[i, t] * (f(v[i], params.ϵ) / v[i]) for i in 1:length(x))
    m = sum((1-params.δ)^i * X[i, t] / v[i] for i in 1:length(x))

    return y, m
end


state(1, Vector{Number}(v_0[:,1]), Vector{Number}(v_0[:,3]), Vector{Number}(v_0[:,4]), 1.0, params)


# Control Transition
function control(t::Int, x::Vector{Number}, v::Vector{Number}, c::Vector{Number}, λ::Number, params::NamedTuple)
    α, β, δ, ϵ, σ, g_A, g_Ae, g, R₀ = params.α, params.β, params.δ, params.ϵ, params.σ, params.g_A, params.g_Ae, params.g, params.R₀
    v̄ = [(λ / c[i]^-(σ))^((ϵ - 1)/ϵ) - 1 for i in 1:length(c)]
    V = zeros(length(v), length(v))   
    C = zeros(length(c), length(c))   
    X = zeros(length(x), length(x))
    for i in 1:length(v)
        for j in 1:i
            V[j, i] = v[j]
            C[j, i] = c[j]
            X[j, i] = x[j]
        end
    end

    for i in 1:length(v)
        for j in 1:i
            if V[j, i] < v̄[i]
                V[j, i] = 0
                C[j, i] = 0
                X[j, i] = 0
            end
        end
    end

    SU_t1 = sum((β*(1-δ))^i * (t+1+i <= length(c) ? C[t+1, t+1+i]^(-σ) : 0) for i in 1:(length(c)-1))
    SU_t = sum((β*(1-δ))^i * (t+i <= length(c) ? C[t, t+i]^(-σ) : 0) for i in 1:(length(c)-1))
    SC_t1 = sum((β*(1-δ)*g)^i for i in 1:count(!iszero, V[:, t+1]); init = 0)
    SC_t = sum((β*(1-δ)*g)^i for i in 1:count(!iszero, V[:, t]); init = 0)

    res1 = (f(v[t+1], ϵ) - f_prime(v[t+1], ϵ) * v[t+1])/(f(v[t], ϵ) - f_prime(v[t], ϵ) * v[t]) * (SU_t1)/(SU_t) - g/β * x[t+1]/x[t] *(SC_t1/SC_t)
    res2 = SU_t1/SU_t - (c[t+1]^(-σ) + λ*(g/β)^(t+1) * SC_t1)/ (c[t]^(-σ) + λ*(g/β)^t * SC_t)
    res3 = c[t] - (1-δ) * sum(((1-δ)^i) * X[i,t]*(f(v[i], ϵ)/v[i])  for i in 1:length(x)) - x[t]

    return res1, res2, res3
end

# Define the matrix of residuals
function F(v::Matrix{Number}, λ::Number)
    R = zeros(eltype(v), T, 5)
    R[1, 1] = v[1, 1] - y_0  # Fix the first value of residuals as a difference to k_0
    R[1, 2] = v[1, 2] - m_0  # Fix the first value of residuals as a difference to e_0    
    R[T, 3] = x_ss - v[T, 3] # Fix the last value of residuals as a difference to x_ss
    R[T, 4] = v_ss - v[T, 4] # Fix the last value of residuals as a difference to v_ss
    R[T, 5] = c_ss - v[T, 5] # Fix the last value of residuals as a difference to c_ss

    for t in 1:T-1
        R[t+1, 1], R[t+1, 2] = state(t, Vector{Number}(v[:, 3]), Vector{Number}(v[:, 4]),Vector{Number}(v[:,5]) , λ, params)
        R[t, 3], R[t, 4], R[t, 5] = control(t, Vector{Number}(v[:, 3]), Vector{Number}(v[:, 4]), Vector{Number}(v[:, 5]), λ, params)
    end

    return R
end

F(v_0, 1.0)

#Initial guess 
T = 10
y_0, m_0, x_0, v_0, c_0 = 1.0, 1.0, 1.0, 1.0, 1.0
v_T = y_ss, m_ss, x_ss, v_ss, c_ss, λ_ss
# Create v initial guess as a 100x5 matrix
v_0 = ones(T, 5)  # Initial guess for v (100x5 matrix)
#convert v_0 to a Matrix{Number}
v_0 = Matrix{Number}(v_0)
λ = 1.0        # Initial guess for λ (scalar)

#Test the F function
F(v_0, λ_0)

# Solve using nlsolve
res = nlsolve(x -> F(x, λ), v_0)
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