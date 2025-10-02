#Adjust the path to the environment and utils.jl as needed
script_dir = @__DIR__
env_dir = joinpath(script_dir, "env")
# ---------------------------------------------------------

using Pkg
Pkg.activate(env_dir)
include(joinpath(script_dir, "functions.jl")) 
using LinearAlgebra, Statistics, Random, Plots, GLM, DataFrames



