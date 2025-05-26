# Transport-Informed Gradient Fields for DREAMPlace

Chip placement optimization using optimal transport theory instead of electrostatic force analogies.

## 📄 Project Overview

**[📖 Read our Report](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/Transport_Informed_Gradient_Fields_for_Global_Placement.pdf)**

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

### Current Issues
- **Too slow**: 30-60 seconds per solve, needs to be much faster to even test with DREAMPlace
- **Optimization**: Partially optimized with PyTorch JIT. There are a few more multigrid-related operations that could be sped up. Although we are limited to a Mac CPU, we will exhaust the torch-related tricks at our disposal. If we still require additional performance, we will rewrite sections of the solver in C++.
- **Testing**: One final bit of testing is required to verify that velocity fields correctly interact with fixed obstacles.

## Results

Transport velocity fields correctly show mass movement from source to target distributions:

### Three Gaussians → Uniform Distribution
![Multimodal Transport](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/multimodal_transport_256x256_10_v_cycles.png)

### Single Gaussian → Uniform Distribution  
![Gaussian to Uniform](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_64x64_gauss_to_uniform.png)

### Gaussian → Gaussian Transport
![Two Gaussians](https://github.com/Ghoobaloo/DREAMPlace/blob/master/dreamplace/ops/flow_based_density_potential/figures/transport_visualization_64x64_two_gauss.png)

*Each visualization shows four panels: source density, target density, velocity field (arrows), and transport streamlines.*

## Repository Structure

```
transport_informed_dreamplace/
├── DREAMPlace/                              # DREAMPlace submodule
│   └── dreamplace/ops/flow_based_density_potential/
│       ├── multigrid_poisson_solver.py     # Core solver
│       ├── flow_based_density_overflow.py  # Density computation  
│       ├── flow_based_density_potential.py # DREAMPlace interface
│       ├── Transport_Informed_Gradient_Fields_for_DREAMPlace.pdf  # Project report
│       └── figures/                         # Results
```

## Getting Started
Follow the same steps to download and install DREAMPlace on your system. Then, run the following:

```bash
cd transport_informed_dreamplace/DREAMPlace/dreamplace/ops/flow_based_density_potential/
python multigrid_poisson_solver.py  # Run tests
```