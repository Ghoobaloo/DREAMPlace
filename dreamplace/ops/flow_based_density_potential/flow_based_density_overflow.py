##
# @file   flow_based_density_overflow.py
# @author Bhrugu Bharathi
# @date   May 2025
# @brief  Compute flow-based density overflow.
#         This version returns the signed density difference (ρ - ρ_target) / Δt,
#         without clamping or summation, suitable for continuity equation solvers.
#

import torch
import dreamplace.ops.density_map.density_map as density_map


class FlowBasedDensityOverflow(object):
    """
    @brief Compute signed density overflow map, suitable for use in the RHS of
           continuity-based Poisson equations. This assumes density map from
           fixed cells is precomputed and cached.
    """
    def __init__(self, node_size_x, node_size_y, 
                 xl, yl, xh, yh, 
                 num_bins_x, num_bins_y, 
                 num_movable_nodes, num_terminals, num_filler_nodes,
                 target_density, deterministic_flag):
        """
        Initialize the density map operators.

        Arguments:
            node_size_x, node_size_y: Sizes of all nodes (movable + fixed + filler)
            xl, yl, xh, yh: Bounds of the layout region
            num_bins_x, num_bins_y: Number of bins in each dimension
            num_movable_nodes: Number of movable standard cells/macros
            num_terminals: Number of fixed terminals/macros
            num_filler_nodes: Number of filler cells
            target_density: Target utilization
            deterministic_flag: Whether to use deterministic ops
        """
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.target_density = target_density
        self.deterministic_flag = deterministic_flag

        self.initial_range_list = [[num_movable_nodes, num_movable_nodes + num_terminals]]
        self.range_list = [
            [0, num_movable_nodes],
            [node_size_x.numel() - num_filler_nodes, node_size_x.numel()]
        ]

        self.density_map_op = None  # will be initialized on first forward()


    def compute_flow_rhs(self, pos, time_step=1.0):
        """
        Raw flow RHS and bin-wise density for continuity-based solver use.
        """
        if self.density_map_op is None:
            fixed_density_op = density_map.DensityMap(
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                xl=self.xl,
                yl=self.yl,
                xh=self.xh,
                yh=self.yh,
                num_bins_x=self.num_bins_x,
                num_bins_y=self.num_bins_y,
                range_list=self.initial_range_list,
                deterministic_flag=self.deterministic_flag,
                initial_density_map=None
            )
            fixed_density = fixed_density_op.forward(pos)

            self.density_map_op = density_map.DensityMap(
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                xl=self.xl,
                yl=self.yl,
                xh=self.xh,
                yh=self.yh,
                num_bins_x=self.num_bins_x,
                num_bins_y=self.num_bins_y,
                range_list=self.range_list,
                deterministic_flag=self.deterministic_flag,
                initial_density_map=fixed_density
            )

        total_density_map = self.density_map_op.forward(pos)
        bin_area = (self.xh - self.xl) * (self.yh - self.yl) / (self.num_bins_x * self.num_bins_y)
        rho_current = total_density_map / bin_area
        flow_rhs = -(rho_current - self.target_density) / time_step
        return flow_rhs, rho_current


    def forward(self, pos, time_step=1.0):
        """
        Compatible forward() call for DREAMPlace. Returns:
            - density_cost: scalar overflow (clamped ReLU residual)
            - max_density: maximum bin density
        """
        flow_rhs, rho_current = self.compute_flow_rhs(pos, time_step)
        density_cost = (rho_current - self.target_density).clamp(min=0.0).sum()
        max_density = rho_current.max()
        return density_cost, max_density
