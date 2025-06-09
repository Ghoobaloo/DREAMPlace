##
# @file   flow_based_density_overflow.py
# @author Bhrugu Bharathi
# @date   May 2025
# @brief  Compute flow-based density overflow using electric potential operator.
#         This version uses the same C++ kernel as ElectricOverflow for exact parity.
#

import torch
from torch import nn
import math
import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
from collections import deque



class FlowBasedDensityOverflow(nn.Module):
    """
    @brief Compute signed density overflow map using electric potential operator
           and return a scalar "overflow cost" + "max density" for DREAMPlace.

    This implementation exactly matches ElectricOverflow's density computation
    for complete parity with DREAMPlace's standard approach.
    """

    def __init__(
        self,
        node_size_x,
        node_size_y,
        bin_center_x,
        bin_center_y,
        target_density,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_terminals,
        num_filler_nodes,
        padding,
        deterministic_flag,
        sorted_node_map,
        movable_macro_mask=None
    ):
        super(FlowBasedDensityOverflow, self).__init__()
        
        # Store all parameters exactly as ElectricOverflow does
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        
        self.bin_center_x = bin_center_x
        self.bin_center_y = bin_center_y
        self.target_density = target_density
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.num_filler_nodes = num_filler_nodes
        self.padding = padding
        self.sorted_node_map = sorted_node_map
        self.movable_macro_mask = movable_macro_mask
        
        self.deterministic_flag = deterministic_flag

        self.internal_iteration_counter = 0 # The idea is that every 25 iterations, we should use the next window size for the free region mask. This aligns with our strategy of progressive window expansion to improve the conditioning of our flow-based gradient fields.
        self.fixed_cell_mask = None
        #self.window_schedule = [32, 64, 128, 256, 512]
        self.window_schedule = [128, 256, 512]
        self.schedule_block_size = 75  # Number of iterations per window size change

        self.reset()

    def extract_centered_window_from_mask(self, mask: torch.Tensor, center_x: int, center_y: int, window_size: int):
        """
        Extract a square window of size `window_size` x `window_size` centered at (center_x, center_y)
        from a larger binary mask (e.g., free-space map or obstacle map).

        Args:
            mask (torch.Tensor): 2D tensor representing the full grid or binary mask.
            center_x (int): X-coordinate (row index) of the center pixel.
            center_y (int): Y-coordinate (column index) of the center pixel.
            window_size (int): Desired size of the square window (must be <= mask dimensions).

        Returns:
            window (torch.Tensor): Extracted window region.
            adjusted_center (Tuple[int, int]): Center coordinates within the window.
            top_left_corner (Tuple[int, int]): Absolute top-left corner coordinates in the full mask.
        """
        # If window size matches full mask, return the whole thing and identity mappings
        if window_size == mask.shape[0] and window_size == mask.shape[1]:
            return mask, (center_x, center_y), (0, 0)

        half_size = window_size // 2

        # Clip to avoid going out of bounds
        start_x = max(0, center_x - half_size)
        end_x = min(mask.shape[0], center_x + half_size)
        start_y = max(0, center_y - half_size)
        end_y = min(mask.shape[1], center_y + half_size)

        # Compute center's position relative to the window
        adjusted_center_x = center_x - start_x
        adjusted_center_y = center_y - start_y

        # Extract window and return info for reinsertion into global mask if needed
        window = mask[start_x:end_x, start_y:end_y]
        adjusted_center = (adjusted_center_x, adjusted_center_y)
        top_left_corner = (start_x, start_y)

        return window, adjusted_center, top_left_corner
    
    def get_contiguous_free_region_in_window(self, window: torch.Tensor, center_x: int, center_y: int) -> torch.Tensor:
        """
        Performs a flood-fill (BFS) from (center_x, center_y) to find the contiguous free region
        within a binary window. Pixels with value 0.0 are considered "free", and tagged pixels
        will be set to 2.0 to mark inclusion in the reachable region.

        Args:
            window (torch.Tensor): 2D binary tensor representing free (0.0) and blocked (>0.0) space.
            center_x (int): X-coordinate (row index) of the starting point (must be within a free cell).
            center_y (int): Y-coordinate (column index) of the starting point.

        Returns:
            tagged_region (torch.Tensor): Copy of `window` with all reachable free pixels tagged as 2.0.
        """
        tagged_region = window.detach().clone()  # Work on a copy to preserve original
        visited = torch.zeros_like(window, dtype=torch.bool)
        visited[center_x, center_y] = True  # Mark the starting pixel as visited
        tag_queue = deque([(center_x, center_y)])

        while tag_queue:
            x, y = tag_queue.popleft()
            tagged_region[x, y] = 2.0  # Mark current pixel as part of the free region

            # Cardinal neighbors: up, down, left, right
            neighbors = [
                (x - 1, y),  # Top
                (x + 1, y),  # Bottom
                (x, y - 1),  # Left
                (x, y + 1)   # Right
            ]

            for nx, ny in neighbors:
                # Skip out-of-bounds neighbors
                if not (nx < 0 or nx >= window.shape[0]) and not (ny < 0 or ny >= window.shape[1]):
                    # Add free neighbors to the queue
                    if not visited[nx, ny] and tagged_region[nx, ny] == 0.0:
                        visited[nx, ny] = True
                        tag_queue.append((nx, ny))

        return tagged_region
    
    def get_free_region_mask(self, tagged_region: torch.Tensor, top_left_corner_x: int, top_left_corner_y: int):
        """
        Converts a locally tagged window (with free cells marked as 2.0) into a global boolean mask
        aligned with the full grid. The tagged region is inserted into the appropriate location
        in the global mask based on the top-left corner coordinates.

        Args:
            tagged_region (torch.Tensor): 2D tensor where contiguous free cells have been tagged with 2.0.
            top_left_corner_x (int): X-coordinate (row) of the window's top-left corner in the full grid.
            top_left_corner_y (int): Y-coordinate (column) of the window's top-left corner in the full grid.

        Returns:
            free_region_mask (torch.Tensor): Boolean mask (same shape as full grid) with free region marked as True.
            num_tagged_cells (int): Number of free cells (tagged as 2.0) in the local window.
        """
        # Count how many free cells were tagged (used for mass normalization)
        num_tagged_cells = (tagged_region == 2.0).sum().item()

        # Initialize full-grid boolean mask
        free_region_mask = torch.zeros_like(self.fixed_cell_mask, dtype=torch.bool)

        # Copy the tagged free region into the appropriate location of the global mask
        free_region_mask[
            top_left_corner_x : top_left_corner_x + tagged_region.shape[0],
            top_left_corner_y : top_left_corner_y + tagged_region.shape[1]
        ] = (tagged_region == 2.0)

        return free_region_mask, num_tagged_cells

    def assemble_free_region_masks(self, fixed_cell_mask: torch.Tensor, window_sizes: list):
        """
        Assembles free region masks for each window size in `window_sizes` based on the fixed cell mask.
        This method extracts windows, performs flood-fill to tag free regions, and constructs global masks.

        Args:
            fixed_cell_mask (torch.Tensor): 2D tensor representing the full grid with fixed cells.
            window_sizes (list): List of window sizes to extract and process.

        Returns:
            free_region_masks (dict): Dictionary mapping each window size to its corresponding free region mask.
            num_tagged_cells_per_size (dict): Number of tagged cells for each window size.
        """
        free_region_masks = {}
        num_tagged_cells_per_size = {}

        # Extract the center of the fixed cell mask
        center_x = fixed_cell_mask.shape[0] // 2
        center_y = fixed_cell_mask.shape[1] // 2

        if fixed_cell_mask[center_x, center_y] != 0.0:
            # Try 8 more locations around the center
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            found_free_cell = False
            for dx, dy in offsets:
                if (0 <= center_x + dx < fixed_cell_mask.shape[0] and
                        0 <= center_y + dy < fixed_cell_mask.shape[1] and
                        fixed_cell_mask[center_x + dx, center_y + dy] == 0.0):
                    center_x += dx
                    center_y += dy
                    found_free_cell = True
                    break
            if not found_free_cell:
                raise ValueError("Center of fixed cell mask must be a free cell (0.0) for window extraction.")

        for size in window_sizes:
            # Extract the centered window
            window, adjusted_center, top_left_corner = self.extract_centered_window_from_mask(
                fixed_cell_mask, center_x, center_y, size)

            # Get the contiguous free region within this window
            tagged_region = self.get_contiguous_free_region_in_window(window, adjusted_center[0], adjusted_center[1])

            # Convert tagged region to global mask
            free_region_mask, num_tagged_cells = self.get_free_region_mask(
                tagged_region, top_left_corner[0], top_left_corner[1])
            
            # Store results
            free_region_masks[size] = free_region_mask
            num_tagged_cells_per_size[size] = num_tagged_cells

        return free_region_masks, num_tagged_cells_per_size

    def reset(self):
        """
        Precompute exactly as ElectricOverflow.reset() does:
         1) Clamped sizes and offsets
         2) Area ratios with macro handling
         3) Number of bins
         4) Maximum impacted bins
         5) Padding mask
        """
        sqrt2 = math.sqrt(2)
        
        # Clamped means stretch a cell to bin size
        # clamped = max(bin_size*sqrt2, node_size)
        self.node_size_x_clamped = self.node_size_x.clamp(min=self.bin_size_x * sqrt2)
        self.offset_x = (self.node_size_x - self.node_size_x_clamped).mul(0.5)
        self.node_size_y_clamped = self.node_size_y.clamp(min=self.bin_size_y * sqrt2)
        self.offset_y = (self.node_size_y - self.node_size_y_clamped).mul(0.5)
        
        # Compute area ratios
        node_areas = self.node_size_x * self.node_size_y
        self.ratio = node_areas / (self.node_size_x_clamped * self.node_size_y_clamped)
        
        # Detect movable macros and scale down the density to avoid halos
        self.num_movable_macros = 0
        if self.target_density < 1 and self.movable_macro_mask is not None:
            self.num_movable_macros = self.movable_macro_mask.sum().data.item()
            self.ratio[:self.num_movable_nodes][self.movable_macro_mask] = self.target_density
        
        # Compute number of bins
        self.num_bins_x = int(round((self.xh - self.xl) / self.bin_size_x))
        self.num_bins_y = int(round((self.yh - self.yl) / self.bin_size_y))
        
        # Compute maximum impacted bins for movable nodes
        if self.num_movable_nodes:
            self.num_movable_impacted_bins_x = int(
                ((self.node_size_x[:self.num_movable_nodes].max() +
                  2 * sqrt2 * self.bin_size_x) /
                 self.bin_size_x).ceil().clamp(max=self.num_bins_x))
            self.num_movable_impacted_bins_y = int(
                ((self.node_size_y[:self.num_movable_nodes].max() +
                  2 * sqrt2 * self.bin_size_y) /
                 self.bin_size_y).ceil().clamp(max=self.num_bins_y))
        else:
            self.num_movable_impacted_bins_x = 0
            self.num_movable_impacted_bins_y = 0
            
        # Compute maximum impacted bins for filler nodes
        if self.num_filler_nodes:
            self.num_filler_impacted_bins_x = (
                (self.node_size_x[-self.num_filler_nodes:].max() +
                 2 * sqrt2 * self.bin_size_x) /
                self.bin_size_x).ceil().clamp(max=self.num_bins_x)
            self.num_filler_impacted_bins_y = (
                (self.node_size_y[-self.num_filler_nodes:].max() +
                 2 * sqrt2 * self.bin_size_y) /
                self.bin_size_y).ceil().clamp(max=self.num_bins_y)
        else:
            self.num_filler_impacted_bins_x = 0
            self.num_filler_impacted_bins_y = 0
            
        # Build padding mask
        if self.padding > 0:
            self.padding_mask = torch.ones(self.num_bins_x,
                                           self.num_bins_y,
                                           dtype=torch.uint8,
                                           device=self.node_size_x.device)
            self.padding_mask[self.padding:self.num_bins_x - self.padding,
                              self.padding:self.num_bins_y - self.padding].fill_(0)
        else:
            self.padding_mask = torch.zeros(self.num_bins_x,
                                            self.num_bins_y,
                                            dtype=torch.uint8,
                                            device=self.node_size_x.device)
        
        # Initial density map due to fixed cells
        self.initial_density_map = None

    def compute_initial_density_map(self, pos):
        """
        Compute initial density map from fixed cells exactly as ElectricOverflow does.
        """
        if self.num_terminals == 0:
            num_fixed_impacted_bins_x = 0
            num_fixed_impacted_bins_y = 0
        else:
            max_size_x = self.node_size_x[self.num_movable_nodes:self.
                                          num_movable_nodes +
                                          self.num_terminals].max()
            max_size_y = self.node_size_y[self.num_movable_nodes:self.
                                          num_movable_nodes +
                                          self.num_terminals].max()
            num_fixed_impacted_bins_x = ((max_size_x + self.bin_size_x) /
                                         self.bin_size_x).ceil().clamp(
                                             max=self.num_bins_x)
            num_fixed_impacted_bins_y = ((max_size_y + self.bin_size_y) /
                                         self.bin_size_y).ceil().clamp(
                                             max=self.num_bins_y)
        
        # Always use C++ implementation (no CUDA needed)
        self.initial_density_map = electric_potential_cpp.fixed_density_map(
            pos, self.node_size_x, self.node_size_y, self.bin_center_x,
            self.bin_center_y, self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y, self.num_movable_nodes,
            self.num_terminals, self.num_bins_x, self.num_bins_y,
            num_fixed_impacted_bins_x, num_fixed_impacted_bins_y,
            self.deterministic_flag)
        
        # Scale density of fixed macros
        self.initial_density_map.mul_(self.target_density)

        # Store the fixed cell mask
        self.fixed_cell_mask = (self.initial_density_map != 0.0).float()

        # Assemble free region masks for each window size
        self.free_region_masks, self.num_tagged_cells_per_size = self.assemble_free_region_masks(
            self.fixed_cell_mask, self.window_schedule)
        

    def get_density_map(self, pos):
        """
        Compute the density map from node positions using the same logic as ElectricOverflow.
        
        This is a refactored helper method that contains the common density map computation
        logic used by both forward() and compute_flow_rhs().
        
        Args:
            pos: Tensor of node positions
            
        Returns:
            density_map: 2D tensor of bin densities (area per bin)
        """
        if self.initial_density_map is None:
            self.compute_initial_density_map(pos)

        # Call density_map with exact same parameters as ElectricOverflow
        output = electric_potential_cpp.density_map(
            pos.view(pos.numel()), self.node_size_x_clamped,
            self.node_size_y_clamped, self.offset_x, self.offset_y, self.ratio,
            self.bin_center_x, self.bin_center_y, self.initial_density_map,
            self.target_density, self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y, self.num_movable_nodes,
            self.num_filler_nodes, self.padding, self.num_bins_x, self.num_bins_y,
            self.num_movable_impacted_bins_x, self.num_movable_impacted_bins_y,
            self.num_filler_impacted_bins_x, self.num_filler_impacted_bins_y,
            self.deterministic_flag)

        density_map = output.view([self.num_bins_x, self.num_bins_y])
        
        # Set padding density exactly as ElectricOverflow does
        if self.padding > 0:
            density_map.masked_fill_(self.padding_mask,
                                    self.target_density * self.bin_size_x * self.bin_size_y)
            
        # Return the density map
        return density_map

    def compute_flow_rhs(self, pos):
        """
        Compute the right-hand side (RHS) for the continuity equation based on current density.
        
        This method is specific to the flow-based density approach and calculates the
        density difference (current - target) for use in flow-based solvers.
        
        Args:
            pos: Tensor of node positions
            
        Returns:
            flow_rhs: (ρ_current - ρ_target), centered around zero mean
            rho_current: Current density (area/bin_area)
        """
        # Get the density map using the same logic as ElectricOverflow
        density_map = self.get_density_map(pos)

        # Convert area to density
        bin_area = self.bin_size_x * self.bin_size_y
        rho_current = density_map / bin_area

        # Get the window size based on the internal iteration counter
        window_index = min(self.internal_iteration_counter // self.schedule_block_size, len(self.window_schedule) - 1)
        window_size = self.window_schedule[window_index]

        # Get the free region mask for the current window size
        free_region_mask = self.free_region_masks[window_size]

        # Get the number of tagged cells in this window
        num_tagged_cells = self.num_tagged_cells_per_size[window_size]

        if num_tagged_cells <= 0:
            raise ValueError(
            "No free cells tagged in the current window size. Check your fixed cell mask and window extraction logic."
            )
        
        target_density = rho_current.sum() / num_tagged_cells 
        rho_target = torch.ones_like(rho_current) * target_density

        # Apply the free region mask to the target density
        rho_target[~free_region_mask] = 0.0

        # Assert mass conservation
        assert torch.isclose(rho_current.sum(), rho_target.sum(), rtol=1e-5, atol=1e-7), \
            "Mass conservation failed: sum(rho_current) != sum(rho_target)"

        # Now, we can compute the flow RHS
        flow_rhs = rho_current - rho_target
        # We mask the flow RHS to zero out regions outside the free region (no flow there)
        flow_rhs[~free_region_mask] = 0.0

        # Lastly, update the internal iteration counter
        self.internal_iteration_counter = self.internal_iteration_counter + 1

        return flow_rhs, rho_current, rho_target

    def forward(self, pos):
        """
        Compute density overflow cost and maximum density exactly as ElectricOverflow does.
        
        This maintains compatibility with DREAMPlace's standard approach while using 
        the refactored get_density_map() helper function internally.
        
        Args:
            pos: Tensor of node positions
            
        Returns:
            density_cost: Sum of density overflow across all bins
            max_density: Maximum density value across all bins
        """
        # Compute density map using refactored helper
        density_map = self.get_density_map(pos)
        bin_area = self.bin_size_x * self.bin_size_y
        density_cost = (density_map -
                        self.target_density * bin_area).clamp_(min=0.0).sum().unsqueeze(0)

        return density_cost, density_map.max().unsqueeze(0) / bin_area
