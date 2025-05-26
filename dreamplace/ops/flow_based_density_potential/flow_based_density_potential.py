"""
@file   flow_based_density_potential.py
@author Bhrugu Bharathi
@date   May 2025
@brief  Flow-based density potential solver for placement using multigrid Poisson solver.
        Interface between DREAMPlace and transport-informed gradient computation.
"""

import torch
from torch import nn
from torch.autograd import Function
import logging

# Import the required modules
from flow_based_density_overflow import FlowBasedDensityOverflow
from multigrid_poisson_solver import MultigridPoissonSolver


class FlowBasedDensityPotentialFunction(Function):
    """
    Automatic differentiation function for transport-informed density forces.
    
    Forward pass: Solves weighted Poisson equation to get transport potential
    Backward pass: Interpolates velocity field to node positions as gradients
    """
    
    @staticmethod
    def forward(
        ctx,
        pos,
        flow_overflow_op,
        multigrid_solver,
        bin_size_x, bin_size_y,
        xl, yl,
        num_bins_x, num_bins_y,
        num_movable_nodes,
        time_step,
    ):
        """
        Forward pass: Solve weighted Poisson equation for transport potential.
        
        Args:
            pos: Node positions [2*num_nodes] (x coords, then y coords)
            flow_overflow_op: FlowBasedDensityOverflow instance
            multigrid_solver: MultigridPoissonSolver instance
            bin_size_x, bin_size_y: Grid spacing
            xl, yl: Layout region origin
            num_bins_x, num_bins_y: Grid dimensions
            num_movable_nodes: Number of movable nodes
            time_step: Time discretization parameter
            
        Returns:
            energy: Scalar transport energy (for gradient computation)
        """
        
        # Compute RHS and current density using FlowBasedDensityOverflow
        flow_rhs, rho_current = flow_overflow_op.compute_flow_rhs(pos, time_step)
        
        # Convert flow RHS to Poisson equation RHS
        # flow_rhs = -(ρ_current - ρ_target) / Δt
        # For Poisson equation ∇·(ρ∇φ) = f, we want f = (ρ_current - ρ_target)
        density_overflow = -flow_rhs * time_step
        
        # Use current density as weights for the Poisson equation
        # Ensure weights are positive and well-conditioned
        bin_area = bin_size_x * bin_size_y
        density_weights = rho_current * bin_area
        density_weights = torch.clamp(density_weights, min=1e-6 * bin_area)
        
        # Solve weighted Poisson equation: ∇·(ρ∇φ) = overflow
        try:
            potential_field = multigrid_solver.solve(
                rhs=density_overflow,
                weights=density_weights,
                h_x=bin_size_x,
                h_y=bin_size_y,
                boundary_conditions='dirichlet'  # Prevent mass flow outside chip
            )
        except Exception as e:
            logging.warning(f"Multigrid solver failed: {e}. Using zero potential.")
            potential_field = torch.zeros_like(density_overflow)
        
        # Compute velocity field: v = -∇φ
        velocity_x, velocity_y = compute_velocity_field(potential_field, bin_size_x, bin_size_y)
        
        # Save tensors for backward pass
        ctx.save_for_backward(pos, velocity_x, velocity_y)
        
        # Store scalar parameters on context
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.xl = xl
        ctx.yl = yl
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_movable_nodes = num_movable_nodes
        
        # Return energy based on velocity magnitude (kinetic energy analog)
        energy = 0.5 * (velocity_x.pow(2) + velocity_y.pow(2)).sum()
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients by interpolating velocity field to nodes.
        
        Args:
            grad_output: Gradient w.r.t. output energy
            
        Returns:
            Gradients w.r.t. all forward pass inputs
        """
        pos, velocity_x, velocity_y = ctx.saved_tensors
        
        # Interpolate velocity field to node positions
        grad_pos = interpolate_velocity_to_nodes(
            pos, velocity_x, velocity_y,
            ctx.xl, ctx.yl, ctx.bin_size_x, ctx.bin_size_y,
            ctx.num_bins_x, ctx.num_bins_y, ctx.num_movable_nodes
        )
        
        # Apply chain rule with grad_output
        grad_pos = grad_pos * grad_output
        
        # Return gradients for all forward arguments (None for non-tensor inputs)
        return (
            grad_pos,     # pos
            None,         # flow_overflow_op  
            None,         # multigrid_solver
            None, None,   # bin_size_x, bin_size_y
            None, None,   # xl, yl
            None, None,   # num_bins_x, num_bins_y
            None,         # num_movable_nodes
            None,         # time_step
        )


def compute_velocity_field(potential_field, bin_size_x, bin_size_y):
    """
    Compute velocity field as v = -∇φ using finite differences.
    
    Args:
        potential_field: Transport potential φ [num_bins_x, num_bins_y]
        bin_size_x, bin_size_y: Grid spacing
        
    Returns:
        velocity_x, velocity_y: Velocity components [num_bins_x, num_bins_y]
    """
    nx, ny = potential_field.shape
    velocity_x = torch.zeros_like(potential_field)
    velocity_y = torch.zeros_like(potential_field)
    
    # Interior points: central differences
    velocity_x[1:-1, :] = -(potential_field[2:, :] - potential_field[:-2, :]) / (2 * bin_size_x)
    velocity_y[:, 1:-1] = -(potential_field[:, 2:] - potential_field[:, :-2]) / (2 * bin_size_y)
    
    # Boundary points: one-sided differences
    if nx > 1:
        velocity_x[0, :] = -(potential_field[1, :] - potential_field[0, :]) / bin_size_x
        velocity_x[-1, :] = -(potential_field[-1, :] - potential_field[-2, :]) / bin_size_x
    if ny > 1:
        velocity_y[:, 0] = -(potential_field[:, 1] - potential_field[:, 0]) / bin_size_y
        velocity_y[:, -1] = -(potential_field[:, -1] - potential_field[:, -2]) / bin_size_y
    
    return velocity_x, velocity_y


def interpolate_velocity_to_nodes(pos, velocity_x, velocity_y, xl, yl,
                                 bin_size_x, bin_size_y, num_bins_x, num_bins_y,
                                 num_movable_nodes):
    """
    Interpolate velocity field from grid to node positions using bilinear interpolation.
    
    Args:
        pos: Node positions [2*num_nodes]
        velocity_x, velocity_y: Velocity field on grid
        xl, yl: Layout region origin
        bin_size_x, bin_size_y: Grid spacing
        num_bins_x, num_bins_y: Grid dimensions
        num_movable_nodes: Number of movable nodes (only these get gradients)
        
    Returns:
        grad_pos: Gradients w.r.t. node positions [2*num_nodes]
    """
    num_nodes = pos.size(0) // 2
    grad_pos = torch.zeros_like(pos)
    
    pos_x = pos[:num_nodes]
    pos_y = pos[num_nodes:]
    
    # Only compute gradients for movable nodes
    for i in range(min(num_movable_nodes, num_nodes)):
        # Convert position to grid coordinates
        grid_x = (pos_x[i] - xl) / bin_size_x
        grid_y = (pos_y[i] - yl) / bin_size_y
        
        # Get integer grid indices and fractional parts
        ix = torch.floor(grid_x).long().item()
        iy = torch.floor(grid_y).long().item()
        wx = grid_x - ix
        wy = grid_y - iy
        
        # Bounds checking for bilinear interpolation
        if 0 <= ix < num_bins_x-1 and 0 <= iy < num_bins_y-1:
            # Bilinear interpolation for x-velocity gradient
            grad_pos[i] = (
                (1-wx) * (1-wy) * velocity_x[ix, iy] +
                wx * (1-wy) * velocity_x[ix+1, iy] +
                (1-wx) * wy * velocity_x[ix, iy+1] +
                wx * wy * velocity_x[ix+1, iy+1]
            )
            
            # Bilinear interpolation for y-velocity gradient
            grad_pos[num_nodes + i] = (
                (1-wx) * (1-wy) * velocity_y[ix, iy] +
                wx * (1-wy) * velocity_y[ix+1, iy] +
                (1-wx) * wy * velocity_y[ix, iy+1] +
                wx * wy * velocity_y[ix+1, iy+1]
            )
        elif 0 <= ix < num_bins_x and 0 <= iy < num_bins_y:
            # Fallback: nearest neighbor for boundary cases
            grad_pos[i] = velocity_x[ix, iy]
            grad_pos[num_nodes + i] = velocity_y[ix, iy]
    
    return grad_pos


class FlowBasedDensityPotential(nn.Module):
    """
    Flow-based density potential with multigrid Poisson solver.
    
    This module computes transport-informed density forces for chip placement
    by solving a weighted Poisson equation derived from optimal transport theory.
    
    Mathematical Background:
        1. Compute density overflow: ρ_current - ρ_target
        2. Solve: ∇·(ρ_current ∇φ) = ρ_current - ρ_target
        3. Compute transport velocity: v = -∇φ
        4. Apply velocity as force field to nodes
    """
    
    def __init__(
        self,
        node_size_x, node_size_y,
        target_density,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_movable_nodes, num_terminals, num_filler_nodes,
        deterministic_flag=False,
        multigrid_levels=4,
        multigrid_tolerance=1e-6,
        multigrid_max_iterations=50,
        time_step=1.0
    ):
        """
        Initialize flow-based density potential solver.
        
        Args:
            node_size_x, node_size_y: Node sizes [num_nodes]
            target_density: Target utilization (e.g., 0.8)
            xl, yl, xh, yh: Layout region bounds
            bin_size_x, bin_size_y: Grid spacing
            num_movable_nodes, num_terminals, num_filler_nodes: Node counts
            deterministic_flag: Use deterministic operations
            multigrid_levels: Number of multigrid levels
            multigrid_tolerance: Convergence tolerance
            multigrid_max_iterations: Maximum V-cycles
            time_step: Time discretization parameter
        """
        super().__init__()
        
        # Store parameters as Python scalars for serialization compatibility
        self.target_density = float(target_density)
        self.xl, self.yl = float(xl), float(yl)
        self.xh, self.yh = float(xh), float(yh)
        self.bin_size_x, self.bin_size_y = float(bin_size_x), float(bin_size_y)
        self.num_movable_nodes = int(num_movable_nodes)
        self.num_terminals = int(num_terminals)
        self.num_filler_nodes = int(num_filler_nodes)
        self.time_step = float(time_step)
        
        # Compute grid dimensions
        self.num_bins_x = int(round((self.xh - self.xl) / self.bin_size_x))
        self.num_bins_y = int(round((self.yh - self.yl) / self.bin_size_y))
        
        # Initialize density overflow computation
        self.flow_overflow_op = FlowBasedDensityOverflow(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            xl=xl, yl=yl, xh=xh, yh=yh,
            num_bins_x=self.num_bins_x,
            num_bins_y=self.num_bins_y,
            num_movable_nodes=num_movable_nodes,
            num_terminals=num_terminals,
            num_filler_nodes=num_filler_nodes,
            target_density=target_density,
            deterministic_flag=deterministic_flag
        )
        
        # Initialize multigrid solver
        self.multigrid_solver = MultigridPoissonSolver(
            num_levels=multigrid_levels,
            tolerance=multigrid_tolerance,
            max_iterations=multigrid_max_iterations
        )
        
        logging.info(f"Initialized FlowBasedDensityPotential with grid {self.num_bins_x}×{self.num_bins_y}")
        
    def forward(self, pos):
        """
        Compute transport-informed density energy.
        
        Args:
            pos: Node positions [2*num_nodes]
            
        Returns:
            energy: Scalar transport energy (enables gradient computation)
        """
        return FlowBasedDensityPotentialFunction.apply(
            pos,
            self.flow_overflow_op,
            self.multigrid_solver,
            self.bin_size_x, self.bin_size_y,
            self.xl, self.yl,
            self.num_bins_x, self.num_bins_y,
            self.num_movable_nodes,
            self.time_step
        )
    
    def get_velocity_field(self, pos):
        """
        Get the transport velocity field for visualization/analysis.
        
        Args:
            pos: Node positions [2*num_nodes]
            
        Returns:
            velocity_x, velocity_y: Velocity field components
            potential_field: Transport potential φ
        """
        # Compute RHS and weights
        flow_rhs, rho_current = self.flow_overflow_op.compute_flow_rhs(pos, self.time_step)
        density_overflow = -flow_rhs * self.time_step
        
        bin_area = self.bin_size_x * self.bin_size_y
        density_weights = torch.clamp(rho_current * bin_area, min=1e-6 * bin_area)
        
        # Solve Poisson equation
        potential_field = self.multigrid_solver.solve(
            rhs=density_overflow,
            weights=density_weights,
            h_x=self.bin_size_x,
            h_y=self.bin_size_y,
            boundary_conditions='dirichlet'
        )
        
        # Compute velocity field
        velocity_x, velocity_y = compute_velocity_field(potential_field, self.bin_size_x, self.bin_size_y)
        
        return velocity_x, velocity_y, potential_field