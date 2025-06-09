# Transport-Informed Gradient Fields for DREAMPlace

Chip placement optimization using optimal transport theory instead of electrostatic force analogies.

## 📄 Project Overview

**[📖 Read our Report](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/transport_informed_gradient_fields_for_DREAMPlace.pdf)**

## Method

We replace DREAMPlace's electrostatic density forces with transport-informed gradients. Instead of treating cells like electric charges, we solve for optimal mass transport between current and target density distributions.

The approach solves this weighted Poisson equation:
```
∇ · (ρ₀(x)∇φ(x)) = ρ₀(x) - ρ₁(x)
```

Where `ρ₀` is current density, `ρ₁` is target density, and `v = -∇φ` gives the transport velocity field. We use a geometric multigrid solver for efficiency.

## Implementation Status

### Completed
- [x] Multigrid Poisson solver with verification
- [x] DREAMPlace integration scaffold  
- [x] Transport visualization tools
- [x] Solution optimization to run 512x512 solves in less than 2 seconds
- [x] Progressive Window Expansion for smoother convergence
- [x] Electrostatic warmup and cooldown periods for the first and last 25 iterations

## Visual Results

Transport velocity fields correctly show mass movement from source to target distributions:

### Three Gaussians → Uniform Distribution (256x256)
![Three Gaussians to Uniform](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_256x256_trimodal_gauss_to_uniform_10_v_cycles.png)

### Single Gaussian → Uniform Distribution (64x64)
![Gaussian to Uniform](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_64x64_gauss_to_uniform.png)

### Gaussian → Gaussian Transport (64x64)
![Two Gaussians](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_64x64_two_gauss.png)

### Hexagonal Honeycomb Pattern -> Uniform Distribution (2048x2048)
![Hexagonal Honeycomb to Uniform](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_2048x2048_hexagonal_honeycomb_to_uniform_20_v_cycles_4_smooth.png)

### Three Gaussians -> Concentric Ring Pattern (2048x2048)
![Three Gaussians to Concentric Rings](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_2048x2048_trimodal_gauss_to_concentric_rings_20_v_cycles_smooth_4.png)

### Example of Smooth Field Generation via Progressive Window Expansion (16x16)
![Three Gaussians to Concentric Rings](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_example_of_progressive_window_expansion_on_16x16.png)

*Each visualization shows four panels: source density, target density, velocity field (arrows), and transport streamlines.*

## Repository Structure

```
transport_informed_dreamplace/
├── DREAMPlace/                              # DREAMPlace submodule
│  └── test/train_and_test_input_json        # Provided training and test input JSON files for Placement Initialization
│  └── train_and_test_sample_metadata/       # Contains corresponding lef/def/netlist files for the training and test circuits provided for project evaluation. You will need to unzip these files to use them
│  └── results/                              # Contains all results of Transport-Informed DREAMPlace runs including output terminal logs
│    └── dreamplace/PlaceObj.py                           # Modified to use the new Transport-Informed gradient fields, with electrostatic warmup and cooldown
│    └── dreamplace/ops/flow_based_density_potential/
│       ├── optimized_multi_grid_solver.py     # Updated solver
│       ├── original_multi_grid_solver.py      # Original solver
│       ├── flow_based_density_overflow.py  # Density computation with Progressive Window Expansion implementation
│       ├── flow_based_density_potential.py # DREAMPlace interface
│       ├── transport_informed_gradient_fields_for_DREAMPlace.pdf  # Project report
│       └── figures/                         # Visual Results
```

## Getting Started
Follow the same steps to download and install DREAMPlace on your system. Then, run the following to generate some sample visualizations:

```bash
cd transport_informed_dreamplace/DREAMPlace/dreamplace/ops/flow_based_density_potential/
python original_multi_grid_solver.py  # Run tests on the original implementation
python test_multi_grid_solver.py # Run tests on the faster implementation
```

Using these Transport-Informed gradient fields with electrostatic warmup and cooldown periods is enabled by default, so you can run DREAMPlace as usual:

```bash
python dreamplace/Placer.py test/train_and_test_input_json/asap7_gcd.json > results/output_logs/asap7_gcd.out
``` 

Note that during the installation you will generate an `install' directory, which contains compiled files necessary for running DREAMPlace. You will likely need to navigate to this directory to run the above commands. If you run into pathing issues as a result, just update the `.json' file paths in the `test/train_and_test_input_json` directory to point to the correct locations of your LEF/DEF/netlist files.

If you run into any issues, please don't hesitate to reach out to us via the GitHub issues page or by email. We will be happy to help you.