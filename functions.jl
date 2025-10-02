#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
using LinearAlgebra, Statistics, Random, Plots, FFTW

# Map (x,y) to linear index with periodic BC
@inline idx(x::Int, y::Int, L::Int) = 1 + mod(x-1, L) + L * mod(y-1, L)

# Precompute 4-neighbor (right, left, up, down) for every site on an LxL lattice
function build_neighbors(L::Int)
    N = L * L
    neighbors = Vector{NTuple{4,Int}}(undef, N)
    for y in 1:L, x in 1:L
        i = idx(x, y, L)
        neighbors[i] = (idx(x+1, y, L), idx(x-1, y, L), idx(x, y+1, L), idx(x, y-1, L))
    end
    return neighbors
end

#Initialize spins as ± 1 in a flat vector (cache-frienfly)
function init_spins(L::Int; ordered::Bool=false, rng=Random.default_rng())
    N = L * L
    s = Vector{Int8}(undef, N)
    if ordered
        fill!(s, 1)
    else
        @inbounds for i in 1:N
            s[i] = rand(rng) < 0.5 ? Int8(1) : Int8(-1)
        end
    end
    return s
end

"""
Perform one Wolff cluster update
Returns the cluster size. Flips are done in-place in `s`.

Argeuments:
- spins::Vector{Int8} : +1/-1 spins in a flat vector
- neighbors::Vector{NTuple{4,Int}} : 4-neighbor list from `build_neighbors`
- K ::Float64 : J/(k_B T)
- visited::BitVector : workspace (length N), reused between calls
- queue::Vector{Int} : workspace queue, reused between calls
- rng::AbstractRNG : random number generator
"""

function wolff_update!( spins::Vector{Int8},
                        neighbors::Vector{NTuple{4,Int}},
                        K::Float64;
                        visited::BitVector,
                        queue::Vector{Int},
                        rng::AbstractRNG=Random.default_rng())
    N = length(spins)
    @assert length(neighbors) == N
    @assert length(visited) == N

    p = 1 - exp(-2K)
    # pick a random seed spin
    seed = rand(rng, 1:N)
    target = spins[seed]

    # reset just the cluster region, not the whole visited array
    # We clear as we go using 'cluster_sites' for speed
    cluster_sites = Int[]
    empty!(cluster_sites)
    empty!(queue)

    #start cluster from seed
    visited[seed] = true
    push!(queue, seed)
    push!(cluster_sites, seed)

    #BFS/DFS via explicit stack/queue
    while !isempty(queue)
        u = pop!(queue) # LIFO = DFS; switch to popfirst! for BFS
        nu = neighbors[u]
        @inbounds for k in 1:4
            v = @inbounds nu[k]
            if !visited[v] && (spins[v] == target)
                if rand(rng) < p
                    visited[v] = true
                    push!(queue, v)
                    push!(cluster_sites, v)
                end
            end
        end
    end

    # flip cluster
    @inbounds for v in cluster_sites
        spins[v] = -spins[v]
        visited[v] = false # reset for next call
    end
    return length(cluster_sites)
end

# Function to find energy per site with J=1
function energy_per_site(spins::Vector{Int8}, neighbors::Vector{NTuple{4,Int}})
    N = length(spins)
    acc = 0
    @inbounds for i in 1:N
        si = spins[i]
        # count right and up only to avoid double counting
        acc += si * spins[neighbors[i][1]] # right
        acc += si * spins[neighbors[i][3]] # up
    end
    return -acc / N
end

# Function to find magnetization per site
function magnetization_per_site(spins::Vector{Int8})
    return mean(spins)
end

# Function to compute the Magnetic Susceptibility
function magnetic_susceptibility(Ms::Vector{Float64}, N::Int)
    mean_M = mean(abs.(Ms))
    mean_M2 = mean(Ms .^ 2)
    chi = N*(mean_M2 - mean_M^2)
    return chi
end

# Function to compute the Specific Heat
function specific_heat(Es::Vector{Float64}, N::Int)
    mean_E = mean(Es)
    mean_E2 = mean(Es .^ 2)
    C = N*(mean_E2 - mean_E^2)
    return C
end

# Function to find two-point Correlation function
function two_point_correlation(spins::Vector{Int8}, L::Int)
    s = reshape(spins, L, L)
    C = zeros(Float64, L)
    for t in 0:L-1
        row0 = s[:, 1]
        rowt = s[:, mod(t,L)+1]
        C[t+1] = sum(row0 .* rowt) / L^2
    end
    return C
end

"""
m_eff(r) = arccosh((C(r+1) + C(r-1)) / (2 * C(r)))
"""
# Function to compute the Effective Mass
# C is length L with C[1]=r=0, C[2]=r=1, ..., C[L]=r=L-1
function meff_cosh(C::AbstractVector{<:Real})
    L = length(C)
    me = fill(NaN, L)  # we'll fill entries for r=2:(L-1)
    @inbounds for r in 2:(L-1)
        den = 2*C[r]
        if den == 0
            continue
        end
        x = (C[r+1] + C[r-1]) / den
        # numerical safety: x should be ≥ 1; allow tiny deficit
        if x >= 1 - 1e-12
            me[r] = acosh(max(x, 1.0))
        end
    end
    return me
end

# Function to find xi at K in the plateau region of C(t)
function xi_plateau(C::AbstractVector{<:Real}, rmin::Int, rmax::Int)
    L = length(C)
    @assert 1 <= rmin < rmax <= div(L,2)
    plateau = C[rmin:rmax]
    return mean(plateau)
end

