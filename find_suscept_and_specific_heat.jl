#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
include(joinpath(script_dir, "functions.jl")) 
using LinearAlgebra, Statistics, Random,Plots, GLM, DataFrames

function find_suscept_and_specific_heat(L::Int, K::Float64; Tstep::Int=10000, equil_steps::Int=5000)
    neighbors = build_neighbors(L)  # Precompute neighbors
    spins = init_spins(L, ordered=false) # Initialize random spins
    N = L * L # Number of spins
    visited = falses(N) # BitVector for visited sites
    queue = Int[] # Queue for cluster growth
    rng = MersenneTwister(1234) # Reproducible RNG
    Es = zeros(Tstep)
    Ms = zeros(Tstep)
    
    for t in 1:Tstep
        wolff_update!(spins,neighbors, K; visited=visited, queue=queue, rng=rng)
        if t > equil_steps
            Es[t] = energy_per_site(spins, neighbors)
            Ms[t] = magnetization_per_site(spins)
        end
    end

    Es = Es[equil_steps+1:end][:]
    Ms = Ms[equil_steps+1:end][:]

    suscept = magnetic_susceptibility(Ms, N)
    specific_heat_K = specific_heat(Es, K)

    return specific_heat_K, suscept
end

function main()
    L_values = [16, 24, 32, 48, 64, 96, 128] # Lattice sizes
    Kc = 0.5*log(1 + sqrt(2)) # Critical K value
    K_values = [i*0.01*Kc for i in 1:130] # K values from 0.01*Kc to 1.4*Kc
    results = Dict()

    for L in L_values
        χs = Float64[]
        Cs = Float64[]
        for K in K_values
            specific_heat, suscept = find_suscept_and_specific_heat(L, K)
            push!(χs, suscept)
            push!(Cs, specific_heat)
        end
        results[L] = (Cs, χs)
    end

    # Plotting
    p1 = plot(title="Specific Heat vs K", xlabel="K", ylabel="Specific Heat", legend=:topleft)
    p2 = plot(title="Magnetic Susceptibility vs K", xlabel="K", ylabel="Magnetic Susceptibility", legend=:topright)
    for L in L_values
        Cs, χs = results[L]
        plot!(p1, K_values, Cs, label="L=$L")
        plot!(p2, K_values, χs, label="L=$L")
    end
    plot(p1, p2, layout=(1,2), size=(900,400), dpi=250, margin=5Plots.mm)
    savefig(joinpath(script_dir, "results/suscept_and_specific_heat.pdf"))

    
    #Plot Kc vs L^(-1)
    L_inv = [1/L for L in L_values]
    Kc_estimates = Float64[]
    for L in L_values
        _, Cs = results[L]
        max_index = argmax(Cs)
        push!(Kc_estimates, K_values[max_index])
    end
    p3 = scatter(L_inv, Kc_estimates .- Kc, xlabel="1/L", ylabel="Kc(L) - Kc ", title="Kc(L) - Kc vs 1/L", legend=false)
    df = DataFrame(x = L_inv, y = Kc_estimates)
    model = lm(@formula(y ~ x), df)
    xnew = range(minimum(L_inv), maximum(L_inv), length=100)
    dfnew = DataFrame(x = xnew)
    yhat = predict(model, dfnew)  # predicted values
    plot!(p3, xnew, yhat .- Kc, label="Linear Fit", color=:red, dpi=250)
    savefig(p3,joinpath(script_dir, "results/Kc_vs_Linv.pdf"))

    #Plot chi_max vs L^(7/4)
    L_pow = [L^(7/4) for L in L_values]
    chi_max_values = Float64[]
    for L in L_values
        _, χs = results[L]
        push!(chi_max_values, maximum(χs))
    end
    p4 = scatter(L_pow, chi_max_values, xlabel="L^(7/4)", ylabel="Max Magnetic Susceptibility", title="Max Magnetic Susceptibility vs L^(7/4)", legend=false)
    df = DataFrame(x = L_pow, y = chi_max_values)
    model = lm(@formula(y ~ x), df)
    xnew = range(minimum(L_pow), maximum(L_pow), length=100)
    dfnew = DataFrame(x = xnew)
    yhat = predict(model, dfnew)  # predicted values
    plot!(p4, xnew, yhat, label="Linear Fit", color=:red, dpi=250)
    savefig(p4, joinpath(script_dir, "results/chi_max_vs_Lpow.pdf"))

    #Plot C_max vs log(L)
    L_log = [log(L) for L in L_values]
    C_max_values = Float64[]
    for L in L_values
        Cs, _ = results[L]
        push!(C_max_values, maximum(Cs))
    end
    p5 = scatter(L_log, C_max_values, xlabel="log(L)", ylabel="Max Specific Heat", title="Max Specific Heat vs log(L)", legend=false)
    df = DataFrame(x = L_log, y = C_max_values)
    model = lm(@formula(y ~ x), df)
    xnew = range(minimum(L_log), maximum(L_log), length=100)
    dfnew = DataFrame(x = xnew)
    yhat = predict(model, dfnew)  # predicted values
    plot!(p5, xnew, yhat, label="Linear Fit", color=:red, dpi=250)
    savefig(p5, joinpath(script_dir, "results/C_max_vs_logL.pdf"))

    return nothing
end

main()
