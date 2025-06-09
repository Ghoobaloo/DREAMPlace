##
# @file   test_multi_grid_solver.py
# @author Bhrugu Bharathi
# @date   May 2025
# @brief  Standalone multigrid solver for weighted Poisson equation
#

import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from typing import Literal

import torch.nn.functional as F

_kernel = torch.tensor([[1,2,1],
                        [2,4,2],
                        [1,2,1]], dtype=torch.float32).div_(16.0).view(1,1,3,3)

_kernel_cache = {}

def _get_kernel(t: torch.Tensor):
    key = (t.dtype, t.device)
    if key not in _kernel_cache:
        _kernel_cache[key] = _kernel.to(dtype=t.dtype, device=t.device)
    return _kernel_cache[key]

def _make_masks(nx, ny, device):
    idx = (torch.arange(nx, device=device).view(-1,1) +
           torch.arange(ny, device=device)) & 1
    red   = (idx == 0)
    black = ~red
    return red, black

# @profile
def apply_bc(phi: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'dirichlet':
        phi[0, :] = 0
        phi[-1, :] = 0
        phi[:, 0] = 0
        phi[:, -1] = 0
    elif mode == 'neumann':
        phi[0, :]  = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0]  = phi[:, 1]
        phi[:, -1] = phi[:, -2]
    else:
        raise ValueError(f"Unknown boundary condition mode: {mode}")
    return phi

# @profile
@torch.jit.script
def smooth_jit_vectorized(
    phi: torch.Tensor,
    rhs: torch.Tensor,
    rho_e: torch.Tensor,
    rho_w: torch.Tensor,
    rho_n: torch.Tensor,
    rho_s: torch.Tensor,
    center_inv: torch.Tensor,
    red_mask: torch.Tensor,
    black_mask: torch.Tensor,
    h_x: float,
    h_y: float,
    num_iterations: int,
    boundary_conditions: str = "dirichlet",
    omega: float = 1.2
) -> torch.Tensor:
    """
    Optimized Red-Black Gauss-Seidel SOR with:
    - red/black masks passed in (precomputed once)
    - harmonic weights precomputed outside sweep loop
    """
    dx2 = h_x * h_x
    dy2 = h_y * h_y

    for _ in range(num_iterations):
        phi = apply_bc(phi, boundary_conditions)

        # RED sweep
        pe = torch.roll(phi, -1, 0)
        pw = torch.roll(phi, 1, 0)
        pn = torch.roll(phi, -1, 1)
        ps = torch.roll(phi, 1, 1)

        neigh = (rho_e * pe + rho_w * pw) / dx2 + (rho_n * pn + rho_s * ps) / dy2
        phi_new = (neigh + rhs) * center_inv
        phi = torch.where(red_mask, phi + omega * (phi_new - phi), phi)

        phi = apply_bc(phi, boundary_conditions)

        # BLACK sweep
        pe = torch.roll(phi, -1, 0)
        pw = torch.roll(phi, 1, 0)
        pn = torch.roll(phi, -1, 1)
        ps = torch.roll(phi, 1, 1)

        neigh = (rho_e * pe + rho_w * pw) / dx2 + (rho_n * pn + rho_s * ps) / dy2
        phi_new = (neigh + rhs) * center_inv
        phi = torch.where(black_mask, phi + omega * (phi_new - phi), phi)

        phi = apply_bc(phi, boundary_conditions)

    return phi

# @profile
def _residual_vec_fast(phi,
                       rhs,
                       rho_e, rho_w, rho_n, rho_s,
                       dx2: float, dy2: float):
    """
    Residual r = f + ∇·(ρ∇φ) using pre-computed face weights.
    Operates entirely with tensor slices—no rolls, no divisions.
    """
    # interior slices of φ (centre and four neighbours)
    c  = phi[1:-1, 1:-1]
    ce = phi[2:,   1:-1]
    cw = phi[:-2,  1:-1]
    cn = phi[1:-1, 2:]
    cs = phi[1:-1, :-2]

    # matching interior slices of the harmonic weights
    re = rho_e[1:-1, 1:-1]
    rw = rho_w[1:-1, 1:-1]
    rn = rho_n[1:-1, 1:-1]
    rs = rho_s[1:-1, 1:-1]

    flux = ((re * (ce - c) - rw * (c - cw)) / dx2 +
            (rn * (cn - c) - rs * (c - cs)) / dy2)

    r_int = rhs[1:-1, 1:-1] + flux            # sign choice unchanged
    out = torch.zeros_like(phi)
    out[1:-1, 1:-1] = r_int
    return out


class MultigridPoissonSolver:
    """
    Geometric multigrid solver for weighted Poisson equation: -∇·(ρ∇φ) = f
    Focus on correctness first, optimization later
    """
    def __init__(self, num_levels=4, num_pre_smooth=8, num_post_smooth=8, 
                 tolerance=1e-6, max_iterations=100):
        self.num_levels = num_levels
        self.num_pre_smooth = num_pre_smooth
        self.num_post_smooth = num_post_smooth
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        

    # @profile
    def solve(self, rhs, weights, h_x, h_y, boundary_conditions='dirichlet'):
        """
        Solve weighted Poisson equation: -∇·(ρ∇φ) = f
        
        Args:
            rhs: Right-hand side f(x,y) [nx, ny] 
            weights: Density weights ρ(x,y) [nx, ny]
            h_x, h_y: Grid spacing
            boundary_conditions: 'neumann' or 'dirichlet'
            
        Returns:
            phi: Potential field [nx, ny]
        """
        nx, ny = rhs.shape
        
        # Initialize solution
        phi = torch.zeros_like(rhs)
        
        # Build multigrid hierarchy
        grids = self._build_grid_hierarchy(rhs, weights, h_x, h_y)
        
        # V-cycle iterations
        for iteration in range(self.max_iterations):
            phi_old = phi.clone()
            phi = self._v_cycle(phi, grids, 0, boundary_conditions)
            
            # Check convergence
            residual_norm = self._compute_residual_norm(phi, grids[0])
            if residual_norm < self.tolerance:
                print(f"Converged in {iteration+1} iterations, residual: {residual_norm:.2e}")
                break
                
        return phi

    # @profile
    def _build_grid_hierarchy(self, rhs, weights, h_x, h_y):
        """Build coarsened grid hierarchy with red/black masks"""
        grids = []
        current_rhs = rhs
        current_weights = weights
        current_hx, current_hy = h_x, h_y

        for level in range(self.num_levels):
            nx, ny = current_rhs.shape

            # Precompute harmonic weights and inverse center coefficient
            wc = current_weights
            we = torch.roll(wc, -1, 0)
            ww = torch.roll(wc, 1, 0)
            wn = torch.roll(wc, -1, 1)
            ws = torch.roll(wc, 1, 1)

            rho_e = 2.0 * wc * we / (wc + we + 1e-12)
            rho_w = 2.0 * wc * ww / (wc + ww + 1e-12)
            rho_n = 2.0 * wc * wn / (wc + wn + 1e-12)
            rho_s = 2.0 * wc * ws / (wc + ws + 1e-12)

            dx2 = current_hx * current_hx
            dy2 = current_hy * current_hy
            center = (rho_e + rho_w) / dx2 + (rho_n + rho_s) / dy2
            center_inv = torch.where(center > 1e-12, 1.0 / center, torch.zeros_like(center))

            red_mask, black_mask = _make_masks(nx, ny, device=current_rhs.device)

            grid = {
                'rhs': current_rhs,
                'rho_e': rho_e,
                'rho_w': rho_w,
                'rho_n': rho_n,
                'rho_s': rho_s,
                'center_inv': center_inv,
                'weights': current_weights,
                'h_x': current_hx,
                'h_y': current_hy,
                'nx': nx,
                'ny': ny,
                'red_mask': red_mask,
                'black_mask': black_mask,
            }
            grids.append(grid)

            if nx <= 4 or ny <= 4:
                break

            # Downsample for next coarser level
            current_rhs = self._restrict(current_rhs)
            current_weights = self._restrict(current_weights)
            current_hx *= 2
            current_hy *= 2

        grid_sizes = [f"{g['nx']}x{g['ny']}" for g in grids]
        print(f"Built {len(grids)} levels: {grid_sizes}")
        return grids

    # @profile
    def _v_cycle(self, phi, grids, level, boundary_conditions):
        """Execute one V-cycle"""
        if level == len(grids) - 1:
            # Coarsest grid - exact solve or many iterations to machine precision
            return self._coarse_solve(phi, grids[level], boundary_conditions)
        
        # Pre-smoothing
        grid = grids[level]
        phi = smooth_jit_vectorized(
            phi,
            grid['rhs'],
            grid['rho_e'], grid['rho_w'], grid['rho_n'], grid['rho_s'], grid['center_inv'],
            grid['red_mask'], grid['black_mask'],
            grid['h_x'], grid['h_y'],
            self.num_post_smooth,
            boundary_conditions,
            omega=1.2
        )

        # Compute residual
        residual = self._compute_residual(phi, grids[level])
        
        # Restrict to coarser grid
        coarse_residual = self._restrict(residual)
        coarse_phi = torch.zeros_like(coarse_residual)
        
        # Update coarse grid RHS
        grids[level+1]['rhs'] = coarse_residual
        
        # Recursively solve on coarser grid
        coarse_correction = self._v_cycle(coarse_phi, grids, level+1, boundary_conditions)
        
        # Prolongate correction back to fine grid
        correction = self._prolongate(coarse_correction, grids[level]['nx'], grids[level]['ny'])
        phi += correction
        
        # Apply boundary conditions after prolongation
        phi = apply_bc(phi, boundary_conditions)
        
        # Post-smoothing
        phi = smooth_jit_vectorized(
            phi,
            grid['rhs'],
            grid['rho_e'], grid['rho_w'], grid['rho_n'], grid['rho_s'], grid['center_inv'],
            grid['red_mask'], grid['black_mask'],
            grid['h_x'], grid['h_y'],
            self.num_pre_smooth,
            boundary_conditions,
            omega=1.2
        )
        
        return phi

    # @profile
    def _coarse_solve(self, phi, grid, boundary_conditions):
        """Direct solve for small coarse grid"""
        nx, ny = grid['nx'], grid['ny']
        rhs = grid['rhs']
        weights = grid['weights']
        h_x, h_y = grid['h_x'], grid['h_y']

        if nx <= 4 and ny <= 4:
            # Build matrix A and flatten rhs for Ax = b
            dx2 = h_x * h_x
            dy2 = h_y * h_y

            n = nx * ny
            A = torch.zeros(n, n, device=phi.device)
            b = rhs.flatten()

            def idx(i, j): return i * ny + j

            for i in range(nx):
                for j in range(ny):
                    k = idx(i, j)
                    if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                        A[k, k] = 1.0  # Dirichlet BC
                        b[k] = 0.0
                    else:
                        rho_e = 2 * weights[i, j] * weights[i + 1, j] / (weights[i, j] + weights[i + 1, j] + 1e-12)
                        rho_w = 2 * weights[i, j] * weights[i - 1, j] / (weights[i, j] + weights[i - 1, j] + 1e-12)
                        rho_n = 2 * weights[i, j] * weights[i, j + 1] / (weights[i, j] + weights[i, j + 1] + 1e-12)
                        rho_s = 2 * weights[i, j] * weights[i, j - 1] / (weights[i, j] + weights[i, j - 1] + 1e-12)

                        center = (rho_e + rho_w) / dx2 + (rho_n + rho_s) / dy2

                        A[k, idx(i + 1, j)] = -rho_e / dx2
                        A[k, idx(i - 1, j)] = -rho_w / dx2
                        A[k, idx(i, j + 1)] = -rho_n / dy2
                        A[k, idx(i, j - 1)] = -rho_s / dy2
                        A[k, k] = center

            try:
                x = torch.linalg.solve(A, b.unsqueeze(1)).squeeze(1)
            except AttributeError:
                x, _ = torch.solve(b.unsqueeze(1), A)
                x = x.squeeze(1)

            return x.view(nx, ny)

        else:
            # Fallback to smoothing
            return self._smooth(phi, grid, 100, boundary_conditions)

    # @profile
    def _compute_residual(self, phi, grid):
        dx2 = grid['h_x'] ** 2
        dy2 = grid['h_y'] ** 2
        return _residual_vec_fast(phi, 
                             grid['rhs'], 
                             grid['rho_e'], grid['rho_w'], grid['rho_n'], grid['rho_s'],
                             dx2, dy2)
    
    # @profile
    def _compute_residual_norm(self, phi, grid):
        """Compute L2 norm of residual"""
        residual = self._compute_residual(phi, grid)
        return torch.norm(residual).item()

    # @profile
    def _restrict(self, fine_grid):
        # kernel = _kernel.to(dtype=fine_grid.dtype, device=fine_grid.device)
        x = fine_grid.unsqueeze(0).unsqueeze(0)   # N,C,H,W
        kernel = _get_kernel(fine_grid)
        x = F.conv2d(x, kernel, stride=2, padding=1)
        return x[0, 0]

    # @profile
    def _prolongate(self, coarse, H, W):
        return torch.nn.functional.interpolate(
            coarse.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )[0,0]


def test_basic_poisson():
    """Test basic Poisson solver functionality"""
    
    print("=== Basic Multigrid Poisson Solver Test ===")
    
    # Test 1: Very simple case - small grid
    print("\n--- Test 1: Simple 8x8 grid, constant RHS ---")
    nx, ny = 8, 8
    hx, hy = 1.0/(nx - 1), 1.0/(ny - 1)
    
    # For -∇²φ = 1 with φ=0 on boundary, expect parabolic solution
    # Maximum should be around 1/(8π²) ≈ 0.0127 for unit square
    rhs = torch.ones(nx, ny) * 1.0
    weights = torch.ones_like(rhs)
    
    solver = MultigridPoissonSolver(num_levels=2, tolerance=1e-8, max_iterations=20)
    phi = solver.solve(rhs, weights, hx, hy, 'dirichlet')
    
    print(f"Solution range: [{phi.min():.6f}, {phi.max():.6f}]")
    print(f"Solution norm: {torch.norm(phi):.6f}")
    print(f"Center value: {phi[nx//2, ny//2]:.6f}")
    print(f"Expected max ≈ 0.0703125 for -∇²φ = 1")
    
    # Test 2: Zero RHS (should give zero solution)
    print("\n--- Test 2: Zero RHS ---")
    rhs_zero = torch.zeros_like(rhs)
    phi_zero = solver.solve(rhs_zero, weights, hx, hy, 'dirichlet')
    zero_error = torch.norm(phi_zero)
    print(f"Zero RHS error: {zero_error:.2e}")
    
    # Test 3: Simple analytical solution verification
    print("\n--- Test 3: Single-mode sine function ---")
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    # For -∇²φ = sin(πx)sin(πy), exact solution is φ = sin(πx)sin(πy)/(2π²)
    rhs_sine = torch.sin(math.pi * X) * torch.sin(math.pi * Y)
    phi_sine = solver.solve(rhs_sine, weights, hx, hy, 'dirichlet')
    
    # Analytical solution 
    phi_exact = rhs_sine / (2 * math.pi**2)
    
    print(f"RHS max: {rhs_sine.max():.6f}")
    print(f"Computed center: {phi_sine[nx//2, ny//2]:.6f}")
    print(f"Exact center: {phi_exact[nx//2, ny//2]:.6f}")
    print(f"Computed max: {phi_sine.max():.6f}")
    print(f"Exact max: {phi_exact.max():.6f}")
    
    if torch.norm(phi_exact) > 1e-10:
        error = torch.norm(phi_sine - phi_exact) / torch.norm(phi_exact)
        print(f"Relative error: {error:.2e}")
        
        # Also check pointwise error at center
        center_error = abs(phi_sine[nx//2, ny//2] - phi_exact[nx//2, ny//2]) / phi_exact[nx//2, ny//2]
        print(f"Center point error: {center_error:.2e}")
        
        return error < 5e-2  # Allow 5% error for 8x8 grid
    else:
        print("Exact solution too small")
        return False

def test_convergence_study():
    """Study convergence with different parameters"""
    
    print("\n=== Convergence Study ===")
    
    # Fixed problem
    nx, ny = 16, 16
    hx, hy = 1.0/(nx - 1), 1.0/(ny - 1)
    
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    rhs = torch.sin(math.pi * X) * torch.sin(math.pi * Y)
    weights = torch.ones_like(rhs)
    phi_exact = rhs / (2 * math.pi**2)
    
    # Test different numbers of smoothing iterations
    smoothing_options = [1, 2, 4, 8]
    
    for num_smooth in smoothing_options:
        solver = MultigridPoissonSolver(
            num_levels=3, 
            num_pre_smooth=num_smooth, 
            num_post_smooth=num_smooth,
            tolerance=1e-10, 
            max_iterations=30
        )
        
        phi = solver.solve(rhs, weights, hx, hy, 'dirichlet')
        error = torch.norm(phi - phi_exact) / torch.norm(phi_exact)
        
        print(f"Smoothing {num_smooth}: error = {error:.2e}")

def visualize_transport(rho0, rho1, vx, vy, X, Y, skip=8, save_file_name="transport_visualization.png"):
    vmag = torch.sqrt(vx**2 + vy**2)

    # Normalize velocity vectors for visualization
    # max_vmag = vmag.max().item()
    # vx_vis = vx / max_vmag
    # vy_vis = vy / max_vmag
    # vmag_vis = vmag / max_vmag

    # Convert field data to NumPy arrays
    rho0_np = rho0.cpu().numpy()
    rho1_np = rho1.cpu().numpy()
    vx_np = vx.cpu().numpy()
    vy_np = vy.cpu().numpy()
    vmag_np = vmag.cpu().numpy()

    # Get spatial resolution from X, Y shape
    ny, nx = X.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X_plot, Y_plot = np.meshgrid(x, y)  # shape: [ny, nx], 'xy' indexing

    # Downsample for quiver
    Xq = X_plot[::skip, ::skip]
    Yq = Y_plot[::skip, ::skip]
    Vxq = vx_np[::skip, ::skip]
    Vyq = vy_np[::skip, ::skip]
    Vmag_q = vmag_np[::skip, ::skip]

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # ρ₀
    im0 = axes[0].imshow(rho0_np, origin='lower', cmap='Reds', extent=[0, 1, 0, 1])
    axes[0].set_title("Source Density ρ₀")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # ρ₁
    im1 = axes[1].imshow(rho1_np, origin='lower', cmap='Blues', extent=[0, 1, 0, 1])
    axes[1].set_title("Target Density ρ₁")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    # Quiver plot (arrow color = |v|)
    q = axes[2].quiver(
        Xq, Yq, Vxq, Vyq, Vmag_q,
        cmap='inferno', scale=50, width=0.003, pivot='mid'
    )
    axes[2].set_title("Quiver ∇φ (arrow color = |v|)")
    fig.colorbar(q, ax=axes[2], label="|v|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    # Streamplot
    strm = axes[3].streamplot(
        X_plot, Y_plot, vx_np, vy_np,
        color=vmag_np, cmap='magma', linewidth=1.0, density=1.2
    )
    axes[3].set_title("Streamplot of ∇φ")
    fig.colorbar(strm.lines, ax=axes[3], label="|v|")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")

    plt.tight_layout()
    plt.savefig(save_file_name, dpi=300)
    plt.show()

def test_simple_weighted_poisson_transport_case():
    print("=== Weighted Poisson Transport-Informed Descent Test ===")
    
    # Grid size
    nx, ny = 64, 64
    hx, hy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    
    # Coordinate grids
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)

    # Define ρ0 (initial mass distribution): Gaussian bump
    rho0 = torch.exp(-100 * ((X - 0.3)**2 + (Y - 0.3)**2)) + 1e-3 # Avoid zero density for stability

    # Define ρ1 (target distribution): slightly shifted Gaussian
    rho1 = torch.exp(-100 * ((X - 0.7)**2 + (Y - 0.7)**2)) + 1e-3 # Avoid zero density for stability

    # RHS: ρ0 - ρ1 (note sign convention matches ∇·(ρ∇φ) = ρ0 - ρ1)
    rhs = rho0 - rho1

    # Call multigrid solver
    solver = MultigridPoissonSolver(num_levels=6, num_pre_smooth=4, num_post_smooth=4,
                                     tolerance=1e-8, max_iterations=50)
    
    print("Solving weighted Poisson equation...")
    
    phi = solver.solve(rhs, weights=rho0, h_x=hx, h_y=hy, boundary_conditions='dirichlet')

    print("Solution computed.")

    # Compute velocity field v = -∇φ using central differences
    def compute_velocity(phi, h_x, h_y):
        vx = torch.zeros_like(phi)
        vy = torch.zeros_like(phi)
        vx[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * h_x)
        vy[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * h_y)
        return vx, vy

    vx, vy = compute_velocity(phi, hx, hy)

    # Visualize
    visualize_transport(rho0, rho1, vx, vy, X, Y, skip=4)

    # Report simple diagnostics
    print(f"Velocity magnitude: min={torch.norm(torch.stack([vx, vy]), dim=0).min():.3e}, max={torch.norm(torch.stack([vx, vy]), dim=0).max():.3e}")

    return phi, vx, vy

def test_weighted_poisson_transport_case_for_variable_max_iters():
    # In this test, let's examine how the solver behaves with different max iterations
    print("=== Weighted Poisson Transport Test with Variable Max Iterations ===")
    # We'll use the same setup as before but let's make the target density uniform.

    nx, ny = 64, 64
    hx, hy = 1.0 / (nx - 1), 1.0 / (ny - 1)

    # Coordinate grids
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)

    # Define ρ0 (initial mass distribution): Gaussian bump
    rho0 = torch.exp(-100 * ((X - 0.3)**2 + (Y - 0.3)**2)) + 1e-3  # Avoid zero density for stability

    # Define ρ1 (target distribution): uniform distribution
    rho1 = torch.ones_like(rho0) * (rho0.sum() / rho0.numel()) # Uniform target density with same total mass

    # RHS: ρ0 - ρ1 (note sign convention matches ∇·(ρ∇φ) = ρ0 - ρ1)
    rhs = rho0 - rho1

    # Test different max iterations
    max_iterations_list = [10, 20, 30, 40, 50]

    def compute_velocity(phi, h_x, h_y):
        vx = torch.zeros_like(phi)
        vy = torch.zeros_like(phi)
        vx[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * h_x)
        vy[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * h_y)
        return vx, vy

    for max_iterations in max_iterations_list:
        print(f"\nTesting with max_iterations={max_iterations}")
        
        solver = MultigridPoissonSolver(num_levels=6, num_pre_smooth=4, num_post_smooth=4,
                                         tolerance=1e-8, max_iterations=max_iterations)
        
        print("Solving weighted Poisson equation...")

        start = time.time()
        
        phi = solver.solve(rhs, weights=rho0, h_x=hx, h_y=hy, boundary_conditions='dirichlet')
        
        end = time.time()

        elapsed_time = end - start

        print(f"Time taken for max_iterations={max_iterations}: {elapsed_time:.2f} seconds")
        print("Solution computed.")

        # Compute velocity field v = -∇φ using central differences
        vx, vy = compute_velocity(phi, hx, hy)

        # Visualize
        visualize_transport(rho0, rho1, vx, vy, X, Y, skip=4, save_file_name=f"transport_visualization_max_iter_{max_iterations}.png")

        # Report simple diagnostics
        print(f"Velocity magnitude: min={torch.norm(torch.stack([vx, vy]), dim=0).min():.3e}, max={torch.norm(torch.stack([vx, vy]), dim=0).max():.3e}")

def create_rectangular_obstacles(nx=512, ny=512):
    """Create rectangular obstacle configuration."""
    # Create coordinate grids
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    obstacle_mask = torch.zeros_like(X, dtype=torch.bool)
    # Define static rectangular obstacles
    obstacles = [
        {'type': 'rectangle', 'x_min': 0.35, 'x_max': 0.45, 'y_min': 0.4, 'y_max': 0.6},
        {'type': 'rectangle', 'x_min': 0.55, 'x_max': 0.65, 'y_min': 0.1, 'y_max': 0.3}
    ]
    for obstacle in obstacles:
        mask = ((X >= obstacle['x_min']) & (X <= obstacle['x_max']) &
                (Y >= obstacle['y_min']) & (Y <= obstacle['y_max']))
        obstacle_mask |= mask

    obstacle_density = 1e-6
    return obstacle_mask, obstacle_density

def create_circular_obstacles(nx=512, ny=512):
    """Create circular obstacle configuration."""
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    obstacle_mask = torch.zeros_like(X, dtype=torch.bool)
    # Define circular obstacles
    circles = [
        {'center': (0.3, 0.3), 'radius': 0.1},
        {'center': (0.7, 0.7), 'radius': 0.08},
        {'center': (0.7, 0.3), 'radius': 0.06}
    ]
    
    for circle in circles:
        cx, cy = circle['center']
        r = circle['radius']
        dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = dist <= r
        obstacle_mask |= mask
    
    obstacle_density = 1e-6
    return obstacle_mask, obstacle_density

def create_triple_gaussian_distribution(nx=512, ny=512):
    """Create triple Gaussian density distribution."""
    # Create coordinate grids
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)

    # Define three Gaussian centers
    centers = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    gaussian_intensity = 500  # Adjust intensity for visibility

    # Create the density distribution
    rho0 = torch.zeros_like(X)

    for center_x, center_y in centers:
        rho0 += torch.exp(-gaussian_intensity * ((X - center_x)**2 + (Y - center_y)**2))

    # Add a small constant to avoid zero density
    rho0 += 1e-3  # Avoid zero density for stability
    return rho0, X, Y

def create_concentric_rings(nx=512, ny=512):
    """Create a unique patterned density distribution with concentric rings."""
    # Create coordinate grids
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    # Center coordinates
    cx, cy = 0.5, 0.5
    
    # Distance from center
    R = torch.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Create concentric rings with alternating high/low density
    rho0 = torch.zeros_like(X)
    
    # Ring parameters
    ring_spacing = 0.08  # Distance between ring centers
    ring_width = 0.03    # Width of each ring
    # base_density = 0.1
    
    # Create 5 concentric rings with alternating densities
    for i in range(5):
        ring_radius = (i + 1) * ring_spacing
        ring_intensity = 2.0 if i % 2 == 0 else 1.0  # Alternate intensities
        
        # Create ring mask using smooth transitions
        ring_mask = torch.exp(-((R - ring_radius) / ring_width)**2 * 50)
        rho0 += ring_intensity * ring_mask
    
    # Add background density and ensure positivity
    rho0 += 1e-3
    # rho0 = torch.clamp(rho0, min=1e-6)
    
    return rho0, X, Y

def create_hexagonal_honeycomb_pattern(nx=512, ny=512):
    """Create hexagonal honeycomb pattern with varying densities."""
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)  # Use explicit indexing
    
    rho0 = torch.zeros_like(X)
    
    # Hexagonal grid parameters
    hex_size = 0.08
    sqrt3 = np.sqrt(3)
    
    # Create hexagonal pattern
    for i in range(-2, 8):
        for j in range(-2, 8):
            # Hexagonal grid coordinates
            hex_x = i * hex_size * 1.5
            hex_y = j * hex_size * sqrt3
            if i % 2 == 1:
                hex_y += hex_size * sqrt3 / 2
            
            center_x = 0.1 + hex_x
            center_y = 0.1 + hex_y
            
            # Skip if outside domain
            if center_x < -0.1 or center_x > 1.1 or center_y < -0.1 or center_y > 1.1:
                continue
            
            # Distance to hexagon center
            dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # Hexagon intensity varies with position - use numpy for scalar math
            intensity = 2.0 * (1 + 0.5 * np.sin(10 * center_x) * np.cos(8 * center_y))
            
            # Create hexagon-like shape (using smoother gaussian for better results)
            hex_mask = torch.exp(-(dist / (hex_size * 0.4))**2 * 10)
            rho0 += intensity * hex_mask
    
    # Add radial modulation
    R = torch.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    radial_mod = 1 + 0.3 * torch.cos(10 * R)
    rho0 *= radial_mod
    
    # Background and positivity
    rho0 += 0.1
    rho0 = torch.clamp(rho0, min=1e-6)
    
    return rho0, X, Y

def create_uniform_distribution(nx=512, ny=512, density=1.0):
    """Create uniform density distribution."""
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    rho = torch.full_like(X, density)
    return rho, X, Y

def create_single_gaussian_distribution(nx=512, ny=512, center=(0.5, 0.5), intensity=100):
    """Create single Gaussian density distribution."""
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    cx, cy = center
    rho = torch.exp(-intensity * ((X - cx)**2 + (Y - cy)**2))
    rho += 1e-6  # Stability
    
    return rho, X, Y

def create_checkerboard_distribution(nx=512, ny=512, squares=8):
    """Create checkerboard pattern density distribution."""
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)
    
    # Create checkerboard pattern
    checker_x = torch.floor(X * squares).int()
    checker_y = torch.floor(Y * squares).int()
    
    # Alternate between high and low density
    rho = torch.where((checker_x + checker_y) % 2 == 0, 2.0, 0.5)
    rho += 0.1  # Base density
    
    return rho, X, Y

# Define distribution dictionaries
OBSTACLE_CONFIGURATIONS = {
    'rectangular': create_rectangular_obstacles,
    'circular': create_circular_obstacles,
    None: lambda nx, ny: (None, None)  # No obstacles
}

SOURCE_DISTRIBUTIONS = {
    'triple_gaussian': create_triple_gaussian_distribution,
    'concentric_rings': create_concentric_rings,
    'hexagonal_honeycomb': create_hexagonal_honeycomb_pattern,
    'single_gaussian': create_single_gaussian_distribution,
    'checkerboard': create_checkerboard_distribution,
    'uniform': create_uniform_distribution
}

TARGET_DISTRIBUTIONS = {
    'uniform': create_uniform_distribution,
    'single_gaussian': create_single_gaussian_distribution,
    'triple_gaussian': create_triple_gaussian_distribution,
    'concentric_rings': create_concentric_rings,
    'checkerboard': create_checkerboard_distribution
}

def compute_velocity(phi, h_x, h_y, obstacle_mask=None):
    """Compute velocity field from potential."""
    vx = torch.zeros_like(phi)
    vy = torch.zeros_like(phi)
    vy[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * h_y)
    vx[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * h_x)

    if obstacle_mask is not None:
        # Set velocity to zero in obstacle regions
        vx[obstacle_mask] = 0.0
        vy[obstacle_mask] = 0.0
    return vx, vy

def test_multimodal_to_uniform_large_grid(
    grid_size=(512, 512),
    density_force_type=Literal['transport', 'electrostatic'],
    obstacle_key=None,
    source_distribution_key='triple_gaussian',
    target_distribution_key='uniform',
    max_iterations_list=None,
    smooth_iters=1,
    save_figures=True,
    figure_prefix="multimodal_transport"
):
    """
    Test transport from source to target distribution with optional obstacles.
    
    Parameters:
    -----------
    grid_size : tuple
        Grid dimensions (nx, ny)
    obstacle_key : str or None
        Key for obstacle configuration. None means no obstacles.
    source_distribution_key : str
        Key for source density distribution
    target_distribution_key : str
        Key for target density distribution
    max_iterations_list : list or None
        List of max iterations to test. Default is [1, 3] if None.
    smooth_iters : int
        Number of smoothing iterations in multigrid solver
    save_figures : bool
        Whether to save visualization figures
    figure_prefix : str
        Prefix for saved figure filenames
    """
    
    # Set default max_iterations_list if not provided
    if max_iterations_list is None:
        max_iterations_list = [1, 3]
    
    nx, ny = grid_size
    print(f"=== Transport: {source_distribution_key} → {target_distribution_key} on {nx}x{ny} Grid ===")
    
    hx, hy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    
    # Get obstacle configuration
    obstacle_func = OBSTACLE_CONFIGURATIONS.get(obstacle_key, OBSTACLE_CONFIGURATIONS[None])
    obstacle_mask, obstacle_density = obstacle_func(nx, ny)
    
    # Get source distribution
    source_func = SOURCE_DISTRIBUTIONS[source_distribution_key]
    rho0, X, Y = source_func(nx, ny)
    
    # Apply obstacles to source if present
    obstacle_flag = obstacle_mask is not None and obstacle_mask.any()

    if obstacle_flag:
        rho0[obstacle_mask] = obstacle_density
    
    # Calculate total mass
    M_total = rho0.sum()
    
    # Get target distribution
    if target_distribution_key == 'uniform':
        # Special handling for uniform target to preserve mass
        if obstacle_flag:
            # With obstacles: preserve low density in obstacles, uniform elsewhere
            free_cells = (~obstacle_mask).sum()
            target_val = (M_total - (obstacle_density * obstacle_mask.sum())) / free_cells
            rho1 = torch.empty_like(rho0)
            rho1[obstacle_mask] = obstacle_density
            rho1[~obstacle_mask] = target_val
        else:
            # No obstacles: uniform density everywhere
            target_val = M_total / (nx * ny)
            rho1 = torch.full_like(rho0, target_val)
    else:
        # Non-uniform target distribution
        target_func = TARGET_DISTRIBUTIONS[target_distribution_key]
        rho1, _, _ = target_func(nx, ny)
        
        # Normalize to preserve total mass
        if obstacle_flag:
            # Apply obstacles to target
            rho1[obstacle_mask] = obstacle_density
            # Normalize free region to preserve mass
            free_mass = M_total - obstacle_density * obstacle_mask.sum()
            current_free_mass = rho1[~obstacle_mask].sum()
            rho1[~obstacle_mask] *= free_mass / current_free_mass
        else:
            # Simple normalization
            rho1 *= M_total / rho1.sum()
    
    # Ensure total mass is preserved
    assert torch.allclose(rho0.sum(), rho1.sum(), rtol=1e-6), \
        f"Total mass mismatch: {rho0.sum()} vs {rho1.sum()}"
    
    print(f"Total mass: {M_total:.6f}")
    print(f"Source density range: [{rho0.min():.3e}, {rho0.max():.3e}]")
    print(f"Target density range: [{rho1.min():.3e}, {rho1.max():.3e}]")
    
    # RHS: ρ₀ - ρ₁
    rhs = (rho0 - rho1)

    if density_force_type == 'transport':
        # Use transport-based forces
        rhoW = rho0
        bc = 'neumann'  # Dirichlet boundary conditions for transport
    elif density_force_type == 'electrostatic':
        rhoW = torch.ones_like(rho0)  # Constant density for Poisson solver
        bc = 'neumann'  # Neumann boundary conditions for electrostatic forces

    for max_iter in max_iterations_list:
        print(f"\n--- Solving with max_iterations={max_iter} ---")
        
        solver = MultigridPoissonSolver(
            num_levels=(nx.bit_length() - 2),  # log2(nx) levels 
            num_pre_smooth=smooth_iters,
            num_post_smooth=smooth_iters,
            tolerance=1e-8,
            max_iterations=max_iter
        )

        start_time = time.time()
        phi = solver.solve(rhs, weights=rhoW, h_x=hx, h_y=hy, boundary_conditions=bc)
        end_time = time.time()

        print(f"Elapsed time: {end_time - start_time:.2f} seconds")

        phi = phi - phi.mean()  # Center potential to avoid drift

        vx, vy = compute_velocity(phi, hx, hy, obstacle_mask=obstacle_mask)

        flux_x = rho0 * vx
        flux_y = rho0 * vy

        # Only compute leakage if there are obstacles
        if obstacle_flag:
            leak_mean = torch.sqrt(flux_x[obstacle_mask]**2 + flux_y[obstacle_mask]**2).mean()
            leak_max = torch.sqrt(flux_x[obstacle_mask]**2 + flux_y[obstacle_mask]**2).max()
            print(f"Leakage in obstacle regions: mean={leak_mean:.4e}, max={leak_max:.4e}")
        else:
            print("No obstacles present - no leakage to compute")

        vmag = torch.sqrt(vx**2 + vy**2)

        print(f"Velocity magnitude stats @ max_iter={max_iter}:")
        print(f"  min: {vmag.min():.4e}")
        print(f"  max: {vmag.max():.4e}")
        print(f"  mean: {vmag.mean():.4e}")
        print(f"  median: {vmag.median():.4e}")

        # Save visualizations if requested
        if save_figures:
            print(f"Visualizing transport for max_iter={max_iter}...")
            filename = f"figures/updated/{figure_prefix}_{source_distribution_key}_to_{target_distribution_key}_{nx}_iter_{smooth_iters}_smooth_{max_iter}_{density_force_type}.png"
            if obstacle_key:
                filename = filename.replace('.png', f'_obstacle_{obstacle_key}.png')
            visualize_transport(rho0, rho1, flux_x, flux_y, X, Y, skip=4, save_file_name=filename)


def gaussian_blur_5x5(input_tensor, sigma=0.1):
    """
    Apply 5x5 Gaussian blur to a 2D input_tensor using a separable convolution.

    Args:
        input_tensor (Tensor): Shape [H, W], the 2D image or density map.
        sigma (float): Standard deviation for Gaussian kernel (controls smoothness).
    
    Returns:
        Tensor: Blurred 2D tensor of shape [H, W]
    """
    # Create 1D Gaussian kernel (5 elements)
    kernel_size = 5
    coords = torch.arange(kernel_size) - kernel_size // 2
    gaussian_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()

    # Create 2D Gaussian by outer product
    gaussian_2d = gaussian_1d[:, None] @ gaussian_1d[None, :]  # shape [5, 5]
    gaussian_2d = gaussian_2d.to(dtype=input_tensor.dtype, device=input_tensor.device)

    # Unsqueeze to apply convolution: [out_channels, in_channels, H, W]
    kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)

    # Add batch and channel dims to input: [B=1, C=1, H, W]
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    # Apply padding and convolution
    blurred = F.conv2d(input_tensor, kernel, padding=2)

    return blurred.squeeze(0).squeeze(0)
            
if __name__ == "__main__":
    # Run basic functionality test
    success = test_basic_poisson()
    
    if success:
        print("\n✓ Basic tests PASSED")
        # test_convergence_study()
    else:
        print("\n✗ Basic tests FAILED")
        print("Focus on fixing fundamental stability issues first")

    # # Run transport test
    # test_simple_weighted_poisson_transport_case()
    # print("\n✓ Transport test completed")
    # Run transport test with variable max iterations
    # test_weighted_poisson_transport_case_for_variable_max_iters()
    # Run multimodal to uniform test on large grid

    test_multimodal_to_uniform_large_grid(
            grid_size=(2048, 2048),
            density_force_type='transport',
            obstacle_key=None,
            source_distribution_key='hexagonal_honeycomb',
            target_distribution_key='triple_gaussian',
            max_iterations_list=[4],
            smooth_iters=4,
            figure_prefix="transport"
        )

    test_multimodal_to_uniform_large_grid(
            grid_size=(2048, 2048),
            density_force_type='electrostatic',
            obstacle_key=None,
            source_distribution_key='hexagonal_honeycomb',
            target_distribution_key='uniform',
            max_iterations_list=[20],
            smooth_iters=4,
            figure_prefix="electrostatic"
        )

