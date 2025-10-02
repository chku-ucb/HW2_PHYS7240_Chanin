#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
using LinearAlgebra, Statistics, Random, Plots

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
    chi = (mean_M2 - mean_M^2)
    return chi
end

# Function to compute the Specific Heat
function specific_heat(Es::Vector{Float64}, K::Float64)
    mean_E = mean(Es)
    mean_E2 = mean(Es .^ 2)
    C = K^2 * (mean_E2 - mean_E^2)
    return C
end

# Function to find Collrelation Function
function correlation_function(spins::Vector{Int8}, L::Int)
    max_dist = div(L, 2) # Maximum distance to consider
    corr = zeros(Float64, max_dist) # Correlation values
    counts = zeros(Int, max_dist) # Count of pairs for normalization
    N = L * L # Total number of spins 
    for y1 in 1:L, x1 in 1:L
        i = idx(x1, y1, L) # Linear index
        s1 = spins[i] # Spin at (x1, y1)
        @inbounds for d in 1:max_dist
            # Horizontal neighbor
            x2 = mod(x1 - 1 + d, L) + 1 # Wrap around using periodic BC
            y2 = y1
            j = idx(x2, y2, L)
            s2 = spins[j]
            corr[d] += s1 * s2
            counts[d] += 1

            # Vertical neighbor
            x2 = x1
            y2 = mod(y1 - 1 + d, L) + 1
            j = idx(x2, y2, L)
            s2 = spins[j]
            corr[d] += s1 * s2
            counts[d] += 1
        end
    end
    # Normalize
    for d in 1:max_dist
        if counts[d] > 0
            corr[d] /= counts[d]
        end
    end
    return corr
end