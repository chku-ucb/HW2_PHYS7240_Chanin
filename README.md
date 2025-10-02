# HW2_PHYS7240_Chanin
This is for code to do HW2 in class Advanced Statistical Mechanic of University of Colorado at Boulder

# How to run code

## Setup
1. Clone the repository to your local machine
2. Open Julia and navigate to the project directory
3. Activate the project environment by running:
    ```julia
    using Pkg
    Pkg.activate("env")
    Pkg.instantiate()
    ```
4. Ensure all dependencies are installed by running:
    ```julia
    Pkg.resolve()
    ```
## Running the Simulation
To run the simulation, execute the following command in Julia:
```julia
julia find_susceptibility.jl
```

# Output
The simulation will generate output files in the `results` directory, including plots and data files for analysis.

# Information for each file
- `functions.jl`: Contains utility functions for the simulation, including index mapping and correlation calculations
- `find_susceptibility.jl`: Main script to run the susceptibility simulation
- `correlations_simulation.jl`: Script to analyze and plot correlation functions from the simulation
- `validation_MandE.jl`: Script to validate magnetization and energy calculations against known results