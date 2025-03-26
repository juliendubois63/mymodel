using Plots

# Load both models inside separate modules to prevent variable conflicts
module BaselineModel
    include("C:/Users/juliendubois/Documents/mymodel/Baseline_HKO_Stacked.jl")
end

module PuttyClayModel
    include("C:/Users/juliendubois/Documents/mymodel/puttyclay_stacked.jl")
end

# Extract results from the baseline model
ces_output = BaselineModel.y_adj
ces_consumption = BaselineModel.v_adj[:, 3]
ces_energy = BaselineModel.v_adj[:, 2]

# Extract results from the putty-clay model
putty_output = PuttyClayModel.v_adj[:, 1]
putty_consumption = PuttyClayModel.v_adj[:, 5]
putty_energy = PuttyClayModel.v_adj[:, 2]

# Create a comparison plot
p1 = plot(ces_output, label = "Baseline Output", title = "Output Comparison", linewidth=2)
plot!(p1, putty_output, label = "Putty-Clay Output", linewidth=2, linestyle=:dash)

p2 = plot(ces_consumption, label = "Baseline Consumption", title = "Consumption Comparison", linewidth=2)
plot!(p2, putty_consumption, label = "Putty-Clay Consumption", linewidth=2, linestyle=:dash)

p3 = plot(ces_energy, label = "Baseline Energy", title = "Energy Use Comparison", linewidth=2)
plot!(p3, putty_energy, label = "Putty-Clay Energy", linewidth=2, linestyle=:dash)

plot(p1, p2, p3, layout = (3, 1))


#Compute the difference between the two models
output_diff = ces_output .- putty_output
consumption_diff = ((ces_consumption .- putty_consumption) ./ ces_consumption) .* 100
consumption_diff = filter(!isnan, consumption_diff)  # Remove NaN values
plot(1:length(consumption_diff), consumption_diff, label="Consumption Difference (%)", title="Consumption Difference")  # Remove Inf values
