#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
include(joinpath(script_dir, "functions.jl")) 
using LinearAlgebra, Statistics, Random, Distributions, Plots

function validation_histogram(L::Int, K::Float64) # Input as lattice length on each side and K value
    neighbors = build_neighbors(L)  # Precompute neighbors
    spins = init_spins(L, ordered=false) # Initialize random spins
    N = L * L # Number of spins
    visited = falses(N) # BitVector for visited sites
    queue = Int[] # Queue for cluster growth
    rng = MersenneTwister(1234) # Reproducible RNG
    Tstep = 10000 # Total Wolff steps
    Es = zeros(Tstep)
    Ms = zeros(Tstep)
    
    for t in 1:Tstep
        wolff_update!(spins,neighbors, K; visited=visited, queue=queue, rng=rng)
        if t > Int(0.5*Tstep)
            Es[t] = energy_per_site(spins, neighbors)
            Ms[t] = magnetization_per_site(spins)
        end
    end

    Es = Es[Int(0.5*Tstep)+1:end][:]
    Ms = Ms[Int(0.5*Tstep)+1:end][:]

    # Plot histograms
    p1 = histogram(Es, bins=30, xlabel="Energy per site", ylabel="Frequency", title="Energy Histogram", legend=false)
    p2 = histogram(Ms, bins=30, xlabel="Magnetization per site", ylabel="Frequency", title="Magnetization Histogram", legend=false)
    plot(p1, p2, layout=(1,2), size=(900,400), dpi=250, margin=5Plots.mm)
    savefig(joinpath(script_dir, "results/validation/validation_histograms_L$(L)_K$(K).pdf"))
    return nothing
end

# Example usage
validation_histogram(24, 0.44) # Near critical temperature
validation_histogram(32, 0.44)  # Near critical temperature
validation_histogram(64, 0.44)  # Near critical temperature
validation_histogram(128, 0.44)  # Near critical temperature

validation_histogram(24, 0.01) # high temperature
validation_histogram(32, 0.01)  # high temperature
validation_histogram(64, 0.01)  # high temperature
validation_histogram(128, 0.01)  # high temperature

validation_histogram(24, 0.5) # low temperature
validation_histogram(32, 0.5)  # low temperature
validation_histogram(64, 0.5)  # low temperature


