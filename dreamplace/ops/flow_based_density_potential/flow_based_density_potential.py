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
import math

# Import the required modules
from dreamplace.ops.flow_based_density_potential.flow_based_density_overflow import FlowBasedDensityOverflow
from dreamplace.ops.flow_based_density_potential.optimized_multi_grid_solver import MultigridPoissonSolver

class FlowBasedDensityPotentialFunction(Function):
    """
    @brief Custom autograd Function for transport-informed density forces.
    
    Forward pass:
      1. call flow_overflow_op.compute_flow_rhs(pos) -> (flow_rhs, rho_current)
      2. build weighted Poisson RHS and weights
      3. call multigrid_solver.solve(...) -> potential_field φ
      4. compute velocity field v = -∇φ via finite differences
      5. store (pos, velocity_x, velocity_y, rho_current) in ctx for backward

    Backward pass:
      1. interpolate (rho_current * velocity) (i.e. “mass flux”) from grid → each node
      2. multiply by grad_output scalar, return as grad_pos
    """

    @staticmethod
    def forward(
        ctx,
        pos,
        flow_overflow_op,
        multigrid_solver,
        bin_center_x,
        bin_center_y,
        node_size_x,
        node_size_y,
        xl, yl,
        xh, yh,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        num_movable_nodes,
        #current_density_weight
    ):
        """
        Args:
            pos:                Tensor of shape [2 * num_nodes], (x coords, then y coords)
            flow_overflow_op:   Instance of FlowBasedDensityOverflow
            multigrid_solver:   Instance of MultigridPoissonSolver
            bin_center_x:       Tensor of shape [num_bins_x]
            bin_center_y:       Tensor of shape [num_bins_y]
            node_size_x:        Tensor of shape [num_nodes], width of each node
            node_size_y:        Tensor of shape [num_nodes], height of each node
            xl, yl, xh, yh:     Floats defining die bounds
            bin_size_x:         Float, bin width in x
            bin_size_y:         Float, bin height in y
            num_bins_x:         Int, number of bins in x direction
            num_bins_y:         Int, number of bins in y direction
            num_movable_nodes:  Int, how many of the first nodes are movable
        Returns:
            energy:  Tensor of shape [1], the transport “energy” = ½ ∑ (ρ·|v|^2)
        """

        # 1) Compute (flow_rhs, rho_current) from overflow operator
        #    - flow_rhs = (ρ_current - ρ_uniform) by construction in FlowBasedDensityOverflow
        #    - rho_current is shape [num_bins_x, num_bins_y] in “mass per area” units
        flow_rhs, rho_current, rho_target = flow_overflow_op.compute_flow_rhs(pos)

        # 2) Build RHS for Poisson: ∇·(ρ_current ∇φ) = (ρ_current - ρ_uniform)
        #    Here `flow_rhs` already equals (ρ_current - ρ_uniform). So we can directly use `flow_rhs`.
        poisson_rhs = flow_rhs  # shape [num_bins_x, num_bins_y]

        # 3) Build density weights for the PDE (the “ρ” in ∇·(ρ ∇φ))
        #    We need weights = ρ_current * bin_area so that the solver works in pure “area” units:
        bin_area = bin_size_x * bin_size_y
        # density_weights = rho_current * bin_area
        # Prevent zero or extremely small weights (for stability):
        density_weights = torch.clamp(rho_current, min=1e-6)# * bin_area)

        # torch.save({'flow_rhs': flow_rhs, 'rho_current': rho_current, 'rho_target': rho_target}, 'dreamplace/ops/flow_based_density_potential/intercepted_tensors_adaptec.pkl')
        # import pdb; pdb.set_trace()

        # 4) Solve ∇·(ρ ∇φ) = poisson_rhs via multigrid:
        #    We expect multigrid_solver.solve(...) to return φ of shape [num_bins_x, num_bins_y].
        #    It should be centered so that mean(φ) = 0 (Neumann BCs).
        try:
            potential_field = multigrid_solver.solve(
                rhs=poisson_rhs,
                weights=density_weights,
                h_x=bin_size_x,
                h_y=bin_size_y,
                boundary_conditions='neumann'  # zero‐flux on edges
            )
        except Exception as e:
            logging.warning(f"Multigrid solver failed: {e}. Returning zero potential.")
            potential_field = torch.zeros_like(poisson_rhs)

        # 5) Compute velocity field v = -∇φ via finite differences.
        #    v has two components (velocity_x, velocity_y), each shape [num_bins_x, num_bins_y].
        velocity_x, velocity_y = compute_velocity_field(potential_field, bin_size_x, bin_size_y)

        # velocity_x = velocity_x * 1e-6  # Scale down to prevent overflow in backward pass
        # velocity_y = velocity_y * 1e-6  # Scale down to prevent overflow in backward pass

        # import pdb; pdb.set_trace()

        # 6) Store for backward:
        #    - We need (pos, rho_current, velocity_x, velocity_y) to compute mass‐flux interpolation.
        ctx.save_for_backward(pos, rho_current, velocity_x, velocity_y)

        # 7) Store scalar/contextual parameters for backward interpolation:
        ctx.bin_center_x = bin_center_x
        ctx.bin_center_y = bin_center_y
        ctx.node_size_x = node_size_x
        ctx.node_size_y = node_size_y
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_movable_nodes = num_movable_nodes

        # 8) Compute a scalar “energy” so that backward( ) is driven by grad(energy) w.r.t. pos.
        #    We choose “kinetic‐energy analog”: ½ ∑_{i,j} ρ_current(i,j) * |v(i,j)|² * bin_area
        #    But since rho_current * bin_area = “mass in bin”, we can do:
        #    energy = 0.5 * ∑ [ (ρ_current * bin_area) * (|v|^2) ] = 0.5 * ∑ [ mass * (|v|^2) ]
        energy = 0.5 * (rho_current * bin_area * (velocity_x.pow(2) + velocity_y.pow(2))).sum()

        # Return a single‐element tensor
        return energy.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: interpolate “mass flux” = (ρ * v) from grid → node positions,
        then multiply by incoming scalar grad_output to produce ∂loss/∂pos.
        """

        # Retrieve saved tensors from forward pass:
        pos, rho_current, velocity_x, velocity_y = ctx.saved_tensors

        # # 1) Form the bin‐wise mass‐flux components: flux_x = (ρ_current * velocity_x * bin_area),
        # #                                           flux_y = (ρ_current * velocity_y * bin_area).
        # #    Because velocity was computed in “per‐unit‐length/sec” and rho_current is “mass per area”,
        # #    mass‐flux (mass/time) per bin = rho_current * velocity * bin_area.
        # bin_area = ctx.bin_size_x * ctx.bin_size_y
        # flux_x = rho_current * velocity_x * bin_area
        # flux_y = rho_current * velocity_y * bin_area

        #NOTE: We'll update velocity_x and velocity_y to be the mass flux directly,

        # 2) Interpolate (velocity_x, velocity_y) from bin grid → each movable node’s position.
        grad_pos = interpolate_flux_to_nodes(
            pos,
            velocity_x,
            velocity_y,
            node_size_x=ctx.node_size_x,
            node_size_y=ctx.node_size_y,
            xl=ctx.xl,
            yl=ctx.yl,
            bin_size_x=ctx.bin_size_x,
            bin_size_y=ctx.bin_size_y,
            num_bins_x=ctx.num_bins_x,
            num_bins_y=ctx.num_bins_y,
            num_movable_nodes=ctx.num_movable_nodes
        )

        # import pdb; pdb.set_trace()

        # 3) Multiply by incoming scalar grad_output (shape [1]) to apply chain rule:
        grad_pos = grad_pos * grad_output.view(1)

        # import pdb; pdb.set_trace()

        # 4) Return gradients for each forward argument.
        #    Only pos has a gradient; everything else is None.
        return (
            grad_pos,  # ∂energy/∂pos
            None,      # flow_overflow_op
            None,      # multigrid_solver
            None, None, None, None, None, None,  # bin_center_x, bin_center_y, node_size_x, node_size_y, xl, yl
            None, None, None, None,  # xh, yh, bin_size_x, bin_size_y
            None, None, None,        # num_bins_x, num_bins_y, num_movable_nodes
            #None       # current_density_weight (not needed in backward)
        )


def compute_velocity_field(potential_field, bin_size_x, bin_size_y):
    """
    Compute velocity field v = -∇φ using 2nd-order finite differences.
    
    Args:
        potential_field: Tensor [num_bins_x, num_bins_y]
        bin_size_x, bin_size_y: Floats
    Returns:
        velocity_x, velocity_y: each Tensor [num_bins_x, num_bins_y]
    """
    nx, ny = potential_field.shape
    velocity_x = torch.zeros_like(potential_field)
    velocity_y = torch.zeros_like(potential_field)

    # Central differences (interior)
    if nx > 2:
        velocity_x[1:-1, :] = -(potential_field[2:, :] - potential_field[:-2, :]) / (2.0 * bin_size_x)
    if ny > 2:
        velocity_y[:, 1:-1] = -(potential_field[:, 2:] - potential_field[:, :-2]) / (2.0 * bin_size_y)

    # One‐sided differences (boundaries)
    if nx > 1:
        velocity_x[0, :] = -(potential_field[1, :] - potential_field[0, :]) / bin_size_x
        velocity_x[-1, :] = -(potential_field[-1, :] - potential_field[-2, :]) / bin_size_x
    if ny > 1:
        velocity_y[:, 0] = -(potential_field[:, 1] - potential_field[:, 0]) / bin_size_y
        velocity_y[:, -1] = -(potential_field[:, -1] - potential_field[:, -2]) / bin_size_y

    return velocity_x, velocity_y

def interpolate_flux_to_nodes(
    pos: torch.Tensor,
    flux_x: torch.Tensor,
    flux_y: torch.Tensor,
    node_size_x: torch.Tensor,
    node_size_y: torch.Tensor,
    xl: float,
    yl: float,
    bin_size_x: float,
    bin_size_y: float,
    num_bins_x: int,
    num_bins_y: int,
    num_movable_nodes: int
) -> torch.Tensor:
    """
    Hybrid interpolation:
      - For “small” nodes (width < bin_size_x AND height < bin_size_y): use bilinear sampling (fully vectorized).
      - For “large” nodes (width >= bin_size_x OR height >= bin_size_y): fall back to exact area‐weighted sum (nested loops).
    """
    device = pos.device
    dtype = pos.dtype

    num_nodes = pos.size(0) // 2
    M = min(int(num_movable_nodes), int(num_nodes))

    # Result (initialized to zero)
    grad_pos = torch.zeros_like(pos, device=device, dtype=dtype)

    # Separate x/y coordinates
    pos_x = pos[:num_nodes]
    pos_y = pos[num_nodes:]

    # 1) Compute bilinear interpolation for ALL nodes in one shot
    #    (We’ll later overwrite the “large” ones with the exact area‐sum result.)
    # --- a) Convert to grid‐space floats: gx = (x – xl)/Δx, gy = (y – yl)/Δy
    gx = (pos_x[:M] - xl) / bin_size_x  # shape [M]
    gy = (pos_y[:M] - yl) / bin_size_y  # shape [M]

    # --- b) floor to get integer base indices, then clamp into [0..num_bins-1]
    ix = torch.floor(gx).long().clamp(0, num_bins_x - 1)   # [M]
    iy = torch.floor(gy).long().clamp(0, num_bins_y - 1)   # [M]

    # --- c) fractional weights w_x = gx - ix, w_y = gy - iy
    wx = (gx - ix.float()).clamp(0.0, 1.0)  # [M]
    wy = (gy - iy.float()).clamp(0.0, 1.0)  # [M]

    # --- d) neighbor indices: ix1 = ix+1 (clamped), iy1 = iy+1 (clamped)
    ix1 = (ix + 1).clamp(0, num_bins_x - 1)
    iy1 = (iy + 1).clamp(0, num_bins_y - 1)

    # --- e) flatten flux → shape [num_bins_x * num_bins_y]
    B = num_bins_y
    flat_fx = flux_x.reshape(-1)  # [num_bins_x * num_bins_y]
    flat_fy = flux_y.reshape(-1)

    # --- f) compute flatten indices for corners (00, 10, 01, 11)
    idx00 = ix * B + iy      # (ix, iy)
    idx10 = ix1 * B + iy     # (ix+1, iy)
    idx01 = ix * B + iy1     # (ix, iy+1)
    idx11 = ix1 * B + iy1    # (ix+1, iy+1)

    # gather corner flux values
    f_x00 = flat_fx[idx00]   # [M]
    f_x10 = flat_fx[idx10]
    f_x01 = flat_fx[idx01]
    f_x11 = flat_fx[idx11]

    f_y00 = flat_fy[idx00]   # [M]
    f_y10 = flat_fy[idx10]
    f_y01 = flat_fy[idx01]
    f_y11 = flat_fy[idx11]

    # --- g) bilinear interpolation formula
    one_m_wx = 1.0 - wx  # [M]
    one_m_wy = 1.0 - wy  # [M]

    bilinear_fx = (one_m_wx * one_m_wy * f_x00
                   + wx       * one_m_wy * f_x10
                   + one_m_wx * wy       * f_x01
                   + wx       * wy       * f_x11)  # [M]

    bilinear_fy = (one_m_wx * one_m_wy * f_y00
                   + wx       * one_m_wy * f_y10
                   + one_m_wx * wy       * f_y01
                   + wx       * wy       * f_y11)  # [M]

    # Write these into grad_pos for all M nodes as a default
    grad_pos[:M] = bilinear_fx
    grad_pos[num_nodes:num_nodes + M] = bilinear_fy

    # 2) Identify which nodes are “large” (≥ 1 bin in either dimension)
    #    For those, we’ll overwrite the bilinear result with an exact area‐weighted sum.
    threshold_x = bin_size_x
    threshold_y = bin_size_y
    large_mask = (node_size_x[:M] >= threshold_x) | (node_size_y[:M] >= threshold_y)
    large_indices = torch.nonzero(large_mask).view(-1)  # e.g. [i0, i1, …]

    # If no large nodes, we’re done.
    if large_indices.numel() == 0:
        return grad_pos

    # 3) For each “large” node i, do the exact area‐weighted sum (nested loops).
    for i in large_indices.tolist():
        # Center and size (pull out as Python floats to keep the loop inner code simple)
        x_center = float(pos_x[i].item())
        y_center = float(pos_y[i].item())
        w = float(node_size_x[i].item())
        h = float(node_size_y[i].item())

        # If degenerate, skip
        if w <= 0.0 or h <= 0.0:
            grad_pos[i] = 0.0
            grad_pos[num_nodes + i] = 0.0
            continue

        # Footprint in world coords
        x_left   = x_center - 0.5 * w
        x_right  = x_center + 0.5 * w
        y_bottom = y_center - 0.5 * h
        y_top    = y_center + 0.5 * h

        # Which bin indices overlap this footprint?
        bin_min_x = int(math.floor((x_left  - xl) / bin_size_x))
        bin_max_x = int(math.floor((x_right - xl) / bin_size_x))
        bin_min_y = int(math.floor((y_bottom - yl) / bin_size_y))
        bin_max_y = int(math.floor((y_top    - yl) / bin_size_y))

        # Clamp into valid range
        bin_min_x = max(bin_min_x, 0)
        bin_max_x = min(bin_max_x, num_bins_x - 1)
        bin_min_y = max(bin_min_y, 0)
        bin_max_y = min(bin_max_y, num_bins_y - 1)

        total_fx = 0.0
        total_fy = 0.0

        # sum over all overlapping bins (double‐loop)
        for bx in range(bin_min_x, bin_max_x + 1):
            # bin’s world X span
            x_bin_left  = xl + bx * bin_size_x
            x_bin_right = x_bin_left + bin_size_x
            overlap_x = min(x_right, x_bin_right) - max(x_left, x_bin_left)
            if overlap_x <= 0.0:
                continue

            for by in range(bin_min_y, bin_max_y + 1):
                # bin’s world Y span
                y_bin_bottom = yl + by * bin_size_y
                y_bin_top    = y_bin_bottom + bin_size_y
                overlap_y = min(y_top, y_bin_top) - max(y_bottom, y_bin_bottom)
                if overlap_y <= 0.0:
                    continue

                overlap_area = overlap_x * overlap_y
                fx_val = float(flux_x[bx, by].item())
                fy_val = float(flux_y[bx, by].item())

                total_fx += fx_val * overlap_area
                total_fy += fy_val * overlap_area

        node_area = w * h
        if node_area > 0.0:
            avg_fx = total_fx / node_area
            avg_fy = total_fy / node_area
        else:
            avg_fx = 0.0
            avg_fy = 0.0

        grad_pos[i] = avg_fx
        grad_pos[num_nodes + i] = avg_fy

    return grad_pos




class FlowBasedDensityPotential(nn.Module):
    """
    Flow-based density potential (multigrid Poisson) module.

    Accepts all the same arguments as the builder function in PlaceObj.py:
        - node_size_x, node_size_y
        - bin_center_x, bin_center_y
        - xl, yl, xh, yh
        - bin_size_x, bin_size_y
        - num_bins_x, num_bins_y
        - num_movable_nodes, num_terminals, num_filler_nodes
        - deterministic_flag
        - movable_macro_mask
        - flow_overflow_op           (instance of FlowBasedDensityOverflow)
        - region_id, fence_regions, node2fence_region_map (currently unused)
        - placedb (unused in this skeleton)
        - name (string, for logging)
    """

    def __init__(
        self,
        node_size_x,
        node_size_y,
        bin_center_x,
        bin_center_y,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_bins_x,
        num_bins_y,
        num_movable_nodes,
        num_terminals,
        num_filler_nodes,
        deterministic_flag=False,
        movable_macro_mask=None,
        flow_overflow_op=None,
        region_id=None,
        fence_regions=None,
        node2fence_region_map=None,
        placedb=None,
        name="FlowDensity"
    ):
        super(FlowBasedDensityPotential, self).__init__()

        # Store everything as Python scalars or tensors as needed
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.bin_center_x = bin_center_x
        self.bin_center_y = bin_center_y

        self.xl = float(xl)
        self.yl = float(yl)
        self.xh = float(xh)
        self.yh = float(yh)
        self.bin_size_x = float(bin_size_x)
        self.bin_size_y = float(bin_size_y)

        self.num_bins_x = int(num_bins_x)
        self.num_bins_y = int(num_bins_y)
        self.num_movable_nodes = int(num_movable_nodes)
        self.num_terminals = int(num_terminals)
        self.num_filler_nodes = int(num_filler_nodes)

        self.deterministic_flag = bool(deterministic_flag)
        self.movable_macro_mask = movable_macro_mask

        # The overflow operator must already be constructed:
        assert isinstance(flow_overflow_op, FlowBasedDensityOverflow), \
            "flow_overflow_op must be a FlowBasedDensityOverflow instance"
        self.flow_overflow_op = flow_overflow_op

        # Keep region/fence placeholders (currently unused)
        self.region_id = region_id
        self.fence_regions = fence_regions
        self.node2fence_region_map = node2fence_region_map
        self.placedb = placedb
        self.name = name

        # Initialize the multigrid solver here (you may fill in details later)
        self.multigrid_solver = MultigridPoissonSolver(
            num_levels=(num_bins_x.bit_length() - 2),  # log2(nx) levels 
            num_pre_smooth=2, 
            num_post_smooth=2, 
            tolerance=1e-6,
            max_iterations=1
        )

        logging.info(f"{name}: FlowBasedDensityPotential initialized on {self.num_bins_x}×{self.num_bins_y} grid")

    def forward(self, pos):#, current_density_weight):
        """
        Compute transport‐informed density energy given node positions `pos`.

        Args:
            pos: Tensor of shape [2 * num_nodes]

        Returns:
            energy: Tensor of shape [1]
        """
        return FlowBasedDensityPotentialFunction.apply(
            pos,
            self.flow_overflow_op,
            self.multigrid_solver,
            self.bin_center_x,
            self.bin_center_y,
            self.node_size_x,
            self.node_size_y,
            self.xl,
            self.yl,
            self.xh,
            self.yh,
            self.bin_size_x,
            self.bin_size_y,
            self.num_bins_x,
            self.num_bins_y,
            self.num_movable_nodes,
            #current_density_weight
        )