# Transport-Informed Gradient Fields for DREAMPlace

Chip placement optimization using optimal transport theory instead of electrostatic force analogies.

## 📄 Project Overview

**[📖 Read our Report](./Transport_Informed_Gradient_Fields_for_DREAMPlace.pdf)**

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
- **Too slow**: 30-60 seconds per solve, needs to be much faster for practical use
- **Optimization**: Partially optimized with PyTorch JIT, may need C++ kernels
- **Testing**: Verifying obstacle awareness and full integration

## Results

Transport velocity fields correctly show mass movement from source to target distributions:

| Scenario | Visualization |
|----------|---------------|
| Three Gaussians → Uniform | ![Multimodal](./dreamplace/ops/flow_based_density_potential/figures/multimodal_transport_256x256_10_v_cycles.png) |
| Gaussian → Uniform | ![Gaussian to Uniform](./dreamplace/ops/flow_based_density_potential/figures/transport_visualization_64x64_gauss_to_uniform.png) |
| Gaussian → Gaussian | ![Two Gaussians](./dreamplace/ops/flow_based_density_potential/figures/transport_visualization_64x64_two_gauss.png) |

*Panels show: source density, target density, velocity field (arrows), transport streamlines.*

## Repository Structure

```
transport_informed_dreamplace/
├── DREAMPlace/                              # DREAMPlace submodule
│   └── dreamplace/ops/flow_based_density_potential/
│       ├── multigrid_poisson_solver.py     # Core solver
│       ├── flow_based_density_overflow.py  # Density computation  
│       ├── flow_based_density_potential.py # DREAMPlace interface
│       └── figures/                         # Results
```

## Getting Started

```bash
git clone --recurse-submodules <repository-url>
cd transport_informed_dreamplace/DREAMPlace/dreamplace/ops/flow_based_density_potential/
python multigrid_poisson_solver.py  # Run tests
```