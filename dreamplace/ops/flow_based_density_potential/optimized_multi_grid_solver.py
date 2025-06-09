"""
Multigrid Poisson Solver for Weighted Poisson Equations

@file   multigrid_poisson_solver.py
@author Bhrugu Bharathi
@date   May 2025
@brief  Optimized multigrid solver for weighted Poisson equation: -∇·(ρ∇φ) = f

This module implements a high-performance geometric multigrid solver for the weighted 
Poisson equation. Key optimizations include pre-computed harmonic weights, vectorized
red-black smoothing, and PyTorch's built-in interpolation for grid transfers.

Mathematical Background:
The weighted Poisson equation is: -∇·(ρ∇φ) = f
where:
- φ is the potential field (solution)
- ρ is the density/weight field  
- f is the right-hand side source term

Performance Optimizations:
- Pre-computed face weights avoid redundant calculations in smoothing
- Cached restriction kernels minimize memory allocations
- Vectorized residual computation using tensor slicing
- PyTorch's optimized conv2d and interpolate for grid transfers
"""

import math
import torch
import numpy as np
import time
from typing import Literal
import torch.nn.functional as F

# Full-weighting restriction kernel (1/16 normalization)
_kernel = torch.tensor([[1,2,1],
                        [2,4,2],
                        [1,2,1]], dtype=torch.float32).div_(16.0).view(1,1,3,3)

# Cache kernels by device/dtype to avoid repeated transfers
_kernel_cache = {}

def _get_kernel(t: torch.Tensor):
    """Get cached restriction kernel matching tensor's device and dtype."""
    key = (t.dtype, t.device)
    if key not in _kernel_cache:
        _kernel_cache[key] = _kernel.to(dtype=t.dtype, device=t.device)
    return _kernel_cache[key]

def _make_masks(nx, ny, device):
    """
    Create red-black checkerboard masks for parallel Gauss-Seidel.
    
    Red points: (i+j) % 2 == 0
    Black points: (i+j) % 2 == 1
    """
    idx = (torch.arange(nx, device=device).view(-1,1) +
           torch.arange(ny, device=device)) & 1
    red   = (idx == 0)
    black = ~red
    return red, black

def apply_bc(phi: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Apply boundary conditions to the potential field.
    
    Args:
        phi: Potential field tensor [nx, ny]
        mode: "dirichlet" (φ=0) or "neumann" (∂φ/∂n=0)
    """
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
    Vectorized Red-Black Gauss-Seidel with SOR acceleration.
    
    Key optimization: All harmonic face weights and the inverse diagonal
    are pre-computed once per grid level, avoiding redundant calculations
    in the smoothing loop.
    
    Args:
        phi: Current solution estimate
        rhs: Right-hand side
        rho_e/w/n/s: Pre-computed harmonic face weights
        center_inv: Pre-computed inverse of diagonal coefficient
        red_mask, black_mask: Pre-computed checkerboard masks
        h_x, h_y: Grid spacing
        num_iterations: Number of smoothing sweeps
        omega: Over-relaxation parameter (1.2 typical)
    """
    dx2 = h_x * h_x
    dy2 = h_y * h_y

    for _ in range(num_iterations):
        phi = apply_bc(phi, boundary_conditions)

        # RED sweep - update all red points simultaneously
        pe = torch.roll(phi, -1, 0)  # East neighbors
        pw = torch.roll(phi, 1, 0)   # West neighbors
        pn = torch.roll(phi, -1, 1)  # North neighbors
        ps = torch.roll(phi, 1, 1)   # South neighbors

        neigh = (rho_e * pe + rho_w * pw) / dx2 + (rho_n * pn + rho_s * ps) / dy2
        phi_new = (neigh + rhs) * center_inv
        phi = torch.where(red_mask, phi + omega * (phi_new - phi), phi)

        phi = apply_bc(phi, boundary_conditions)

        # BLACK sweep - identical to red but with black mask
        pe = torch.roll(phi, -1, 0)
        pw = torch.roll(phi, 1, 0)
        pn = torch.roll(phi, -1, 1)
        ps = torch.roll(phi, 1, 1)

        neigh = (rho_e * pe + rho_w * pw) / dx2 + (rho_n * pn + rho_s * ps) / dy2
        phi_new = (neigh + rhs) * center_inv
        phi = torch.where(black_mask, phi + omega * (phi_new - phi), phi)

        phi = apply_bc(phi, boundary_conditions)

    return phi

def _residual_vec_fast(phi,
                       rhs,
                       rho_e, rho_w, rho_n, rho_s,
                       dx2: float, dy2: float):
    """
    Compute residual r = f + ∇·(ρ∇φ) using tensor slicing.
    
    Avoids torch.roll overhead by directly slicing interior points
    and their neighbors. The flux divergence is computed using
    pre-computed harmonic face weights.
    
    Returns:
        Residual field with zeros on boundaries
    """
    # Interior slices of φ
    c  = phi[1:-1, 1:-1]   # Center
    ce = phi[2:,   1:-1]   # East
    cw = phi[:-2,  1:-1]   # West
    cn = phi[1:-1, 2:]     # North
    cs = phi[1:-1, :-2]    # South

    # Matching interior slices of face weights
    re = rho_e[1:-1, 1:-1]
    rw = rho_w[1:-1, 1:-1]
    rn = rho_n[1:-1, 1:-1]
    rs = rho_s[1:-1, 1:-1]

    # Flux divergence: ∇·(ρ∇φ)
    flux = ((re * (ce - c) - rw * (c - cw)) / dx2 +
            (rn * (cn - c) - rs * (c - cs)) / dy2)

    r_int = rhs[1:-1, 1:-1] + flux
    out = torch.zeros_like(phi)
    out[1:-1, 1:-1] = r_int
    return out


class MultigridPoissonSolver:
    """
    Optimized geometric multigrid solver for weighted Poisson equation.
    
    This implementation prioritizes performance through:
    - Pre-computed harmonic weights at each grid level
    - Vectorized smoothing with cached masks
    - Efficient grid transfers using PyTorch ops
    - Minimal memory allocations in hot paths (via kernel caching)
    """
    
    def __init__(self, num_levels=4, num_pre_smooth=8, num_post_smooth=8, 
                 tolerance=1e-6, max_iterations=100):
        """
        Initialize solver parameters.
        
        Args:
            num_levels: Maximum multigrid levels (limited by grid size)
            num_pre_smooth: Pre-smoothing iterations
            num_post_smooth: Post-smoothing iterations  
            tolerance: Convergence tolerance on residual norm
            max_iterations: Maximum V-cycles
        """
        self.num_levels = num_levels
        self.num_pre_smooth = num_pre_smooth
        self.num_post_smooth = num_post_smooth
        self.tolerance = tolerance
        self.max_iterations = max_iterations

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
        
        # Build multigrid hierarchy with pre-computed weights
        grids = self._build_grid_hierarchy(rhs, weights, h_x, h_y)
        
        # V-cycle iterations
        for iteration in range(self.max_iterations):
            phi_old = phi.clone()
            phi = self._v_cycle(phi, grids, 0, boundary_conditions)
            
            # Check convergence
            # residual_norm = self._compute_residual_norm(phi, grids[0])
            # if residual_norm < self.tolerance:
            #     # print(f"Converged in {iteration+1} iterations, residual: {residual_norm:.2e}")
            #     break

        # Center towards mean
        if boundary_conditions == 'neumann':
            # For Neumann BCs, solution is only defined up to a constant
            # Center to zero mean to ensure uniqueness
            phi = phi - torch.mean(phi)
            
        return phi

    def _build_grid_hierarchy(self, rhs, weights, h_x, h_y):
        """
        Build coarsened grid hierarchy with pre-computed smoothing data.
        
        For each level, pre-computes:
        - Harmonic face weights (rho_e/w/n/s)
        - Inverse diagonal coefficient (center_inv)
        - Red-black masks
        
        This avoids redundant calculations in the smoothing iterations.
        """
        grids = []
        current_rhs = rhs
        current_weights = weights
        current_hx, current_hy = h_x, h_y

        for level in range(self.num_levels):
            nx, ny = current_rhs.shape

            # Pre-compute harmonic averages at cell faces
            wc = current_weights
            we = torch.roll(wc, -1, 0)
            ww = torch.roll(wc, 1, 0)
            wn = torch.roll(wc, -1, 1)
            ws = torch.roll(wc, 1, 1)

            rho_e = 2.0 * wc * we / (wc + we + 1e-12)
            rho_w = 2.0 * wc * ww / (wc + ww + 1e-12)
            rho_n = 2.0 * wc * wn / (wc + wn + 1e-12)
            rho_s = 2.0 * wc * ws / (wc + ws + 1e-12)

            # Pre-compute inverse of diagonal coefficient
            dx2 = current_hx * current_hx
            dy2 = current_hy * current_hy
            center = (rho_e + rho_w) / dx2 + (rho_n + rho_s) / dy2
            center_inv = torch.where(center > 1e-12, 1.0 / center, torch.zeros_like(center))

            # Pre-compute red-black masks
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

            # Coarsen for next level
            current_rhs = self._restrict(current_rhs)
            current_weights = self._restrict(current_weights)
            current_hx *= 2
            current_hy *= 2

        # grid_sizes = [f"{g['nx']}x{g['ny']}" for g in grids]
        # print(f"Built {len(grids)} levels: {grid_sizes}")
        return grids

    def _v_cycle(self, phi, grids, level, boundary_conditions):
        """
        Execute one V-cycle of the multigrid method.
        
        Follows standard V-cycle pattern:
        1. Pre-smooth
        2. Compute and restrict residual
        3. Recursively solve coarse correction
        4. Prolongate and add correction
        5. Post-smooth
        """
        if level == len(grids) - 1:
            # Coarsest grid - solve exactly or to machine precision
            return self._coarse_solve(phi, grids[level], boundary_conditions)
        
        # Pre-smoothing with pre-computed weights
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
        
        # Update coarse grid RHS for correction equation
        grids[level+1]['rhs'] = coarse_residual
        
        # Recursively solve coarse correction
        coarse_correction = self._v_cycle(coarse_phi, grids, level+1, boundary_conditions)
        
        # Prolongate correction to fine grid
        correction = self._prolongate(coarse_correction, grids[level]['nx'], grids[level]['ny'])
        phi += correction
        
        # Apply boundary conditions after correction
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

    def _coarse_solve(self, phi, grid, boundary_conditions):
        """
        Direct solve on coarsest grid.
        
        For very small grids (≤4x4), constructs and solves the full
        linear system. Otherwise falls back to many smoothing iterations.
        """
        nx, ny = grid['nx'], grid['ny']
        rhs = grid['rhs']
        weights = grid['weights']
        h_x, h_y = grid['h_x'], grid['h_y']

        if nx <= 4 and ny <= 4:
            # Build full matrix for direct solve
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
                        # Harmonic averages for matrix entries
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
            # Fallback: many smoothing iterations
            return self._smooth(phi, grid, 100, boundary_conditions)

    def _compute_residual(self, phi, grid):
        """Compute residual using pre-computed face weights."""
        dx2 = grid['h_x'] ** 2
        dy2 = grid['h_y'] ** 2
        return _residual_vec_fast(phi, 
                             grid['rhs'], 
                             grid['rho_e'], grid['rho_w'], grid['rho_n'], grid['rho_s'],
                             dx2, dy2)
    
    def _compute_residual_norm(self, phi, grid):
        """Compute L2 norm of residual for convergence check."""
        residual = self._compute_residual(phi, grid)
        return torch.norm(residual).item()

    def _restrict(self, fine_grid):
        """
        Full-weighting restriction using conv2d.
        
        Uses cached 3x3 kernel with stride 2 for efficient downsampling.
        The kernel implements standard full-weighting stencil.
        """
        x = fine_grid.unsqueeze(0).unsqueeze(0)   # Add batch and channel dims
        kernel = _get_kernel(fine_grid)
        x = F.conv2d(x, kernel, stride=2, padding=1)
        return x[0, 0]

    def _prolongate(self, coarse, H, W):
        """
        Bilinear prolongation using PyTorch's interpolate.
        """
        return torch.nn.functional.interpolate(
            coarse.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )[0,0]