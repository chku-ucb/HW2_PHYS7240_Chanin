#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
include(joinpath(script_dir, "functions.jl")) 
using LinearAlgebra, Statistics, Random, Plots, GLM, DataFrames


function find_two_point_corr_at_K(L::Int, K::Float64; Tstep::Int=10000, equil_steps::Int=5000)
    neighbors = build_neighbors(L)
    spins = init_spins(L, ordered=false)
    N = L * L
    visited = falses(N)
    queue = Int[]
    rng = MersenneTwister(1234)
    avg_corr = zeros(Float64, L) # Store average correlation function
    
    for t in 1:Tstep
        wolff_update!(spins,neighbors, K; visited=visited, queue=queue, rng=rng)
        if t > equil_steps
            # Compute two-point correlation function for this configuration
            corr = two_point_correlation(spins, L)
            avg_corr .+= corr
        end
    end
    avg_corr ./= (Tstep - equil_steps)
    return avg_corr # Return average two-point correlation function
end




function main()
    Kc = 0.5*log(1 + sqrt(2))
    L_values = [16, 24, 32, 48, 64, 96, 128]
    K_values = [i*0.01*Kc for i in 1:130] # K values from 0.01*Kc to 1.3*Kc
    someK = [0.01, 0.25, 0.5, 0.75, 1.0, 1.25] * Kc
    results = Dict()
    for L in L_values
        corrs = []
        for K in K_values
            avg_corr = find_two_point_corr_at_K(L, K)
            push!(corrs, avg_corr)
        end
        results[L] = corrs
    end
    # Plotting
    for i in 1:length(L_values)
        L = L_values[i]
        p = plot(title="Two-Point Correlation Function for L=$L", xlabel="Distance r", ylabel="C(r)", dpi=250)
        corrs = results[L]
        for j in 1:length(K_values)
            K = K_values[j]
            if K in someK
                corr = corrs[j]
                r_vals = collect(-div(L,2):(div(L,2)-1))
                corr_vals = vcat(corr[div(L,2)+1:end], corr[1:div(L,2)])
                if K == Kc
                    plot!(p, r_vals, corr_vals, label="K=$(round(K, digits=3)) (Critical)", lw=2, color=:red)
                else
                    plot!(p, r_vals, corr_vals, label="K=$(round(K, digits=3))")
                end
            else
                continue
            end
        end
        savefig(p, joinpath(script_dir, "results/correlations/two_point_corr_L$(L).pdf"))
    end

    #Plot m_eff(t) vs t for each L and K
    for (i, L) in pairs(L_values)
        p = plot(title="Effective Mass for L=$L",
                xlabel="distance r", ylabel="m_eff(r)", dpi=250, legend=:topright)

        corrs = results[L]      # assume this is a Vector of connected C(r) for each K, length L
        rmin = 2
        rmax = floor(Int, L ÷ 2)           # avoid wrap-around region
        r_vals = collect(rmin:rmax)

        for (j, K) in pairs(K_values)
            if K in someK
                C = corrs[j]                   # length L, C[1]=r=0, C[2]=r=1, ...
                me = meff_cosh(C)              # length L, valid from 2..L-1
                y = @view me[rmin:rmax]
                # drop NaNs (numerical noise near boundaries)
                keep = .!isnan.(y)
                lbl = K ≈ Kc ? "K=$(round(K,digits=3)) (critical)" : "K=$(round(K,digits=3))"
                plot!(p, r_vals[keep], y[keep], label=lbl, lw=(K≈Kc ? 2 : 1.5))
            end
        end

        savefig(p, joinpath(script_dir, "results/correlations/m_eff_L$(L).pdf"))
    end

    # Plot xi(K) vs K for each L and compare to |K-Kc|^-1
    for (i, L) in pairs(L_values)
        p = plot(title="Correlation Length Estimate (ξ) for L=$L",
                 xlabel="K", ylabel="ξ(K)", dpi=250)
        corrs = results[L]
        rmin = 2
        rmax = floor(Int, L ÷ 3)
        xis = Float64[]
        N = L * L

        for (j, K) in pairs(K_values)
            C = corrs[j]
            xi, A, model = xi_find(C, rmin, rmax)
            push!(xis, xi)
        end
        
        plot!(p, K_values, xis, label="ξ(K) (L=$L)", lw=2)

        # Plot |K-Kc|^-1 for K < Kc and near Kc
        delta = 0.1 * Kc  # range near Kc to show asymptotic behavior
        asymp_Ks = [K for K in K_values if (K < Kc) && (Kc - K < delta) && (abs(K - Kc) > 1e-3)]
        asymp_xi = [abs(K - Kc)^(-1) for K in asymp_Ks]
        # Scale for visibility using max(xis) in this region
        scale = maximum(xis)
        asymp_xi_scaled = scale * asymp_xi ./ maximum(asymp_xi)
        plot!(p, asymp_Ks, asymp_xi_scaled, label="scaled |K-Kc|⁻¹ (K < Kc, near Kc)", lw=1.5, linestyle=:dash, color=:black)

        savefig(p, joinpath(script_dir, "results/correlations/xi_L$(L).pdf"))
    end

    return nothing
end

main()