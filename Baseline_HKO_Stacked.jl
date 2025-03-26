using Plots
using NLsolve
using ForwardDiff
using SparseArrays
using LinearAlgebra

# Define parameters
σ = 1.0
ϵ = 0.02
β = 0.985
δ = 0.05
R₀ = 1000
g_A = 1.01

# Adjust β and compute g_Ae
β *= g_A^((1-σ)/2)
g_Ae = g_A^2/ β
g = g_Ae / g_A^2

params = (; β, δ, ϵ, σ, g_A, g_Ae, g, R₀)

# Utility function
function u(c::Number, σ::Number)
    return isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
end

# Utility function and derivative
function u_prime(c::Number, σ::Number)
    c = ForwardDiff.derivative(x -> u(x, σ), c)
    return c
end

# Production function and derivatives
function f(k::Number, e::Number,  ϵ::Number)
    return ((k)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(ϵ/(ϵ-1))
end

function f_prime_k(k::Number, e::Number, ϵ::Number)
    return ForwardDiff.derivative(x -> f(x, e, ϵ), k)
end

function f_prime_e(k::Number, e::Number, ϵ::Number)
    return ForwardDiff.derivative(x -> f(k, x, ϵ), e)
end

# Compute steady-state values
e_ss = ((g - 1) / g) * R₀

function solve_steady_state(e_ss, g_A, β, δ, ϵ, σ)
    function equations(vars)
        k, c = vars
        eq1 = c^σ - (β / g_A^2) * (f_prime_k(k, e_ss, ϵ) + 1 - δ) * c^σ #euler equation at c_t+1 = c_t
        eq2 = k - f(k, e_ss, ϵ) - (1 - δ) * k + c #resource constraint at k_t+1 = k_t
        return [eq1, eq2]
    end
    solution = nlsolve(equations, [10.0, 10.0], show_trace = true)
    return solution.zero
end

k_ss, c_ss = solve_steady_state(e_ss, g_A, β, δ, ϵ, σ)

f(k_ss,e_ss,α,ϵ)

# Transition functions
function state(k::Number, e::Number, c::Number, params)
    return f(k, e, params.ϵ) + (1 - params.δ) * k - c
end

# Define the control function that computes both residuals at the same time
function control(k::Number, k_next::Number, e::Number, e_next::Number, c::Number, c_next::Number, params)
    res1 = u_prime(c, params.σ) * params.g_A^2 - params.β * u_prime(c_next, params.σ) * (f_prime_k(k_next, e_next, params.ϵ) + 1 - params.δ)
    res2 = params.β * u_prime(c_next, params.σ) / u_prime(c, params.σ) * (f_prime_e(k_next, e_next, params.ϵ) / f_prime_e(k, e,  params.ϵ)) - (1 / params.g)
    return res1, res2
end 

# Define the function F
function F(v::Matrix)
    R = zeros(T, 3)
    R[1, 1] = v[1, 1] - k_0  # Fix the first value of residuals as a difference to k_0
    R[T, 2] = e_ss - v[T, 2] # Fix the last value of residuals as a difference to e_ss
    R[T, 3] = c_ss - v[T, 3] # Fix the last value of residuals as a difference to c_ss
    for t in 1:T-1
        k, e, c = v[t, :] # Current values of k, e, c
        k_next = state(k, e, c, params) # Compute the next value of k
        e_next, c_next = v[t+1, 2], v[t+1, 3] # Use the next values of e and c from v
        # Compute the residuals
        R[t+1, 1] = v[t+1, 1] - k_next  
        R[t, 2], R[t, 3] = control(k, v[t+1, 1], e, e_next, c, c_next, params)
    end
    return R
end

# Define F as taking vector u as input in c_values
function F(u::Vector)
    v = reshape(u, 3, T)
    r = F(v)
    r = reshape(r, 3*T)
    return r
end

# Define the vector v
v_T = [k_ss, e_ss, c_ss]
h = 1e-1
k_0, e_0, c_0 = 5.0,13.0,1.0
v_0 = [k_0, e_0, c_0]
T = 100  # Time horizon

#Define v as a linear increase from v_0 to v_T
v = [v_0 + (v_T - v_0) * t / T for t in 0:T-1]

# Define a v with only k_0, random values and v_T
#v = rand(T, 3)
#v[1, 1] = k_0
#v[T, :] .= v_T

# Convert v to a matrix
v = hcat(v...)'
v = Matrix(v)

# Solve using NLsolve
res = nlsolve(x -> F(x), v, show_trace = true)
v_sol = reshape(res.zero, T, 3)

# Plot the results in 3 different graphs
# Define a function that creates a layout 3x1 with the graphs
function plot_results(v::Matrix)
    p1 = plot(v[:, 1], label = "k", title = "Capital")
    p2 = plot(v[:, 2], label = "e", title = "Energy")
    p3 = plot(v[:, 3], label = "c", title = "Consumption")
    plot(p1, p2, p3, layout = (3, 1))
end

# Print the residuals as a matrix T*3 form
v = F(v_sol)

#Adjust the path by multiplying them by g_A^(t/(1-α)) to obtain non stationarized
v_adj = copy(v_sol)
for t in 1:T
    v_adj[t, 1] *= g_A^(2t)
    v_adj[t, 2] /= g^(t - 1)
    v_adj[t, 3] *= g_A^(2t)
end

# Create a function to plot the two paths, stationarized and non-stationarized on a 3x2 layout
function plot_results(v::Matrix, v_adj::Matrix, y::Vector, y_adj::Vector)
    p1 = plot(v[:, 1], label = "k", title = "Capital", titlefontsize=10, linecolor=:blue)
    p2 = plot(v[:, 2], label = "e", title = "Energy", titlefontsize=10, linecolor=:green)
    p3 = plot(v[:, 3], label = "c", title = "Consumption", titlefontsize=10, linecolor=:red)
    p4 = plot(v_adj[:, 1], label = "k", title = "Capital (non-stationarized)", titlefontsize=10, linecolor=:blue)
    p5 = plot(v_adj[:, 2], label = "e", title = "Energy (non-stationarized)", titlefontsize=10, linecolor=:green)
    p6 = plot(v_adj[:, 3], label = "c", title = "Consumption (non-stationarized)", titlefontsize=10, linecolor=:red)
    plot(p1, p4, p2, p5, p3, p6, layout = (3, 2))
end

# Call the function to plot the results
plot_results(v_sol, v_adj, y, y_adj)



#Compute output level, meaning f(k,e) for each t
y = [f(v_sol[t, 1], v_sol[t, 2], ϵ) for t in 1:T]
y_adj = [y[t]* g_A^(2t) for t in 1:T]   

#Compute investment level, meaning f(k,e) - c for each t
x = [y[t] - v_sol[t, 3] for t in 1:T]
x_adj = [x[t] * g_A^(2t) for t in 1:T]

#Plot the output 
function plot_output(y, y_adj, x, x_adj)
    p1 = plot(y, label = "CES", title = "Output", titlefontsize=10, linecolor=:blue)
    p2 = plot(y_adj, label = "CES", title = "Output (non-stationarized)", titlefontsize=10, linecolor=:blue)
    p3 = plot(x, label = "CES", title = "Investment", titlefontsize=10, linecolor=:green)
    p4 = plot(x_adj, label = "CES", title = "Investment (non-stationarized)", titlefontsize=10, linecolor=:green)
    plot(p1, p2, p3, p4, layout = (2, 2))
end

plot_output(y, y_adj, x, x_adj)

#compute and plot growth rate of variables from v_adj
function plot_growth(v_adj::Matrix)
    k_growth = v_adj[2:T, 1] ./ v_adj[1:T-1, 1]
    e_growth = v_adj[2:T, 2] ./ v_adj[1:T-1, 2]
    c_growth = v_adj[2:T, 3] ./ v_adj[1:T-1, 3]
    e_growth = (v_adj[2:T, 2] ./ v_adj[1:T-1, 2]) .- 1
    c_growth = (v_adj[2:T, 3] ./ v_adj[1:T-1, 3]) .- 1
    p1 = plot(k_growth, label = "k", title = "Capital growth rate", titlefontsize=10, linecolor=:blue)
    p2 = plot(e_growth, label = "e", title = "Energy growth rate", titlefontsize=10, linecolor=:green)
    p3 = plot(c_growth, label = "c", title = "Consumption growth rate", titlefontsize=10, linecolor=:red)
    plot(p1, p2, p3, layout = (3, 1))
end
plot_growth(v_adj)

#Graph for export
#extract v_adj values and shape them properly, with nice legend (without non-stationarized) 
#download them, each on a graph separately at a right format
k = v_adj[:, 1]
e = v_adj[:, 2]
c = v_adj[:, 3]

p1 = plot(1:T, k, label = "CES", title = "Capital", titlefontsize=10, linecolor=:blue)
savefig(p1, "C:/Users/juliendubois/Downloads/capital_CES.png")
p2 = plot(1:T, e, label = "CES", title = "Energy", titlefontsize=10, linecolor=:green)
savefig(p2, "C:/Users/juliendubois/Downloads/energy_CES.png")
p3 = plot(1:T, c, label = "CES", title = "Consumption", titlefontsize=10, linecolor=:red)
savefig(p3, "C:/Users/juliendubois/Downloads/consumption_CES.png")


