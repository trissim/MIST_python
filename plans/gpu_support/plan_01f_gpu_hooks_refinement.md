# plan_01f_gpu_hooks_refinement.md
## Component: Translation Refinement Backend Integration

### Objective
Refactor translation_refinement.py to support pluggable backend logic, allowing it to use either CPUBackend or GPUBackend depending on runtime configuration. This will complete the full modularity of the image alignment process and enable GPU acceleration for the computationally intensive hill climbing operations.

### Plan
1. Modify the Refine base class to accept a ComputeBackend instance
   - Add a backend parameter to relevant methods
   - Replace direct PCIAM.compute_cross_correlation calls with backend method calls
   - Ensure backward compatibility with static methods

2. Update hill_climb_worker to use the backend
   - Replace PCIAM.compute_cross_correlation calls with backend.compute_cross_correlation
   - Optionally use backend.hill_climb for a more optimized implementation
   - Maintain the cache mechanism for performance

3. Update multipoint_hill_climb to use the backend
   - Pass the backend to hill_climb_worker
   - Consider using batch operations where appropriate

4. Modify RefineSequential and RefineParallel classes
   - Add backend initialization in constructors using create_compute_backend
   - Pass the backend to optimize_direction and other methods
   - Ensure proper backend handling in parallel processing

5. Update the _worker method in RefineParallel
   - Create a new backend instance for each worker to ensure thread safety
   - Pass the backend to optimize_direction

6. Add logging for backend usage
   - Log which backend is being used for refinement
   - Add timing information for performance comparison

### Findings
The translation_refinement.py module currently has several areas where it directly calls PCIAM static methods, making it CPU-bound and preventing GPU acceleration:

1. **hill_climb_worker**: This method is the core of the refinement process and makes direct calls to `pciam.PCIAM.compute_cross_correlation`. It's called repeatedly during the hill climbing process, making it a prime candidate for GPU acceleration.

```python
best_peak.ncc = pciam.PCIAM.compute_cross_correlation(i1, i2, best_peak.x, best_peak.y)
```

```python
ncc = pciam.PCIAM.compute_cross_correlation(i1, i2, new_x, new_y)
```

2. **multipoint_hill_climb**: This method calls hill_climb_worker multiple times with different starting points, which could benefit from batch processing on the GPU.

3. **optimize_direction**: This method calls multipoint_hill_climb and is the entry point for the refinement process.

4. **RefineParallel._worker**: This method is used for parallel processing and calls optimize_direction, but it doesn't currently have a way to pass a backend instance.

The current architecture doesn't allow for pluggable backends in the translation refinement process, which means it can't take advantage of the GPU acceleration provided by the GPUBackend. By refactoring these components to use the ComputeBackend interface, we can enable GPU acceleration for the computationally intensive hill climbing operations.

Additionally, the GPUBackend already implements a hill_climb method that could potentially replace the hill_climb_worker method in Refine, providing a more optimized implementation for GPU execution. However, we need to ensure that the cache mechanism is maintained for performance.

### Implementation Draft

Here's the implementation for making translation_refinement.py backend-agnostic:

```python
import argparse
import numpy as np
import logging
import time
from abc import ABC
import copy
import enum
import functools

# local imports
import MIST.img_grid as img_grid
import MIST.img_tile as img_tile
import MIST.stage_model as stage_model
import MIST.pciam as pciam
import MIST.utils as utils
from MIST.compute_backend import ComputeBackend
from MIST.backend_factory import create_compute_backend


class HillClimbDirection(enum.Enum):
    """
    Defines hill climbing direction using cartesian coordinates when observing a two dimensional
    grid where the upper left corner is 0,0. Moving north -1 in the y-direction, south +1 in the
    y-direction, west -1 in the x-direction, and east +1 in the x-direction.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    NoMove = (0, 0)

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Refine(ABC):
    @staticmethod
    def hill_climb_worker(i1: np.ndarray, i2: np.ndarray,
                          x_min: int, x_max: int, y_min: int, y_max: int,
                          start_x: int, start_y: int, cache: np.ndarray,
                          backend: ComputeBackend = None) -> img_tile.Peak:
        """
        Computes cross correlation search with hill climbing.

        Args:
            i1: image 1 (ego)
            i2: image 2 (neighbor: north or west)
            x_min: min x boundary
            x_max: max x boundary
            y_min: min y boundary
            y_max: max y boundary
            start_x: start x position for the hill climb
            start_y: start y position for the hill climb
            cache: 2D array of np.float32 storing the NCC values for each (x,y)
            backend: ComputeBackend instance to use for computation (optional)

        Returns:
            A Peak object with the best correlation and its (x, y) position.
        """
        # If no backend is provided, use the legacy PCIAM static methods
        if backend is None:
            # For backward compatibility
            compute_cross_correlation = pciam.PCIAM.compute_cross_correlation
        else:
            compute_cross_correlation = backend.compute_cross_correlation

            # If the backend has a hill_climb method, use it directly
            if hasattr(backend, 'hill_climb') and callable(getattr(backend, 'hill_climb')):
                return backend.hill_climb(i1, i2, start_x, start_y, max(x_max - x_min, y_max - y_min), cache)

        best_peak = img_tile.Peak(ncc=np.nan, x=start_x, y=start_y)

        while True:
            cur_direction = HillClimbDirection.NoMove

            # Translate current absolute position to 0-based cache indices.
            cur_x_idx = best_peak.x - x_min
            cur_y_idx = best_peak.y - y_min

            # Clamp the indices to ensure they fall within cache bounds.
            cur_x_idx = max(0, min(cur_x_idx, cache.shape[1] - 1))
            cur_y_idx = max(0, min(cur_y_idx, cache.shape[0] - 1))

            best_peak.ncc = cache[cur_y_idx, cur_x_idx]
            if np.isnan(best_peak.ncc):
                best_peak.ncc = compute_cross_correlation(i1, i2, best_peak.x, best_peak.y)
                cache[cur_y_idx, cur_x_idx] = best_peak.ncc

            search_center = copy.deepcopy(best_peak)

            # Prepare a list of directions to check
            directions = []
            for d in HillClimbDirection._member_names_:
                dir = HillClimbDirection[d]
                if dir == HillClimbDirection.NoMove:
                    continue

                # Compute new absolute positions.
                new_x = search_center.x + dir.x
                new_y = search_center.y + dir.y

                # Ensure new absolute positions are within search bounds.
                if y_min <= new_y <= y_max and x_min <= new_x <= x_max:
                    # Compute corresponding cache indices and clamp them.
                    new_x_idx = max(0, min(cur_x_idx + dir.x, cache.shape[1] - 1))
                    new_y_idx = max(0, min(cur_y_idx + dir.y, cache.shape[0] - 1))

                    # If the NCC value is not in the cache, add this direction to the list
                    if np.isnan(cache[new_y_idx, new_x_idx]):
                        directions.append((dir, new_x, new_y, new_x_idx, new_y_idx))

            # If we have a backend with batch computation capability and there are multiple directions to check,
            # use batch computation for better performance
            if backend is not None and hasattr(backend, 'batch_compute_cross_correlation') and len(directions) > 1:
                offsets = [(new_x, new_y) for _, new_x, new_y, _, _ in directions]
                ncc_values = backend.batch_compute_cross_correlation(i1, i2, offsets)

                # Update the cache with the computed values
                for i, (_, new_x, new_y, new_x_idx, new_y_idx) in enumerate(directions):
                    cache[new_y_idx, new_x_idx] = ncc_values[i]
            else:
                # Compute NCC values one by one
                for dir, new_x, new_y, new_x_idx, new_y_idx in directions:
                    ncc = compute_cross_correlation(i1, i2, new_x, new_y)
                    cache[new_y_idx, new_x_idx] = ncc

            # Find the best direction
            for d in HillClimbDirection._member_names_:
                dir = HillClimbDirection[d]
                if dir == HillClimbDirection.NoMove:
                    continue

                # Compute new absolute positions.
                new_x = search_center.x + dir.x
                new_y = search_center.y + dir.y

                # Ensure new absolute positions are within search bounds.
                if y_min <= new_y <= y_max and x_min <= new_x <= x_max:
                    # Compute corresponding cache indices and clamp them.
                    new_x_idx = max(0, min(cur_x_idx + dir.x, cache.shape[1] - 1))
                    new_y_idx = max(0, min(cur_y_idx + dir.y, cache.shape[0] - 1))

                    ncc = cache[new_y_idx, new_x_idx]
                    if ncc > best_peak.ncc:
                        best_peak.ncc = ncc
                        best_peak.x = new_x
                        best_peak.y = new_y
                        cur_direction = dir

            if cur_direction == HillClimbDirection.NoMove:
                break

        if np.isnan(best_peak.ncc):
            best_peak.x = int((x_max + x_min) / 2)
            best_peak.y = int((y_max + y_min) / 2)
            best_peak.ncc = -1.0

        return best_peak

    @staticmethod
    def multipoint_hill_climb(num_hill_climbs: int, t1: img_tile.Tile, t2: img_tile.Tile,
                             x_min: int, x_max: int, y_min: int, y_max: int,
                             start_x: int, start_y: int, backend: ComputeBackend = None) -> img_tile.Peak:
        """
        Performs hill climbing from multiple starting points to find the best correlation peak.

        Args:
            num_hill_climbs: Number of hill climbs to perform
            t1: First tile
            t2: Second tile
            x_min: Minimum x boundary
            x_max: Maximum x boundary
            y_min: Minimum y boundary
            y_max: Maximum y boundary
            start_x: Initial x position
            start_y: Initial y position
            backend: ComputeBackend instance to use for computation (optional)

        Returns:
            The Peak with the highest NCC value
        """
        start_time = time.time()

        i1 = t1.get_image()
        i2 = t2.get_image()
        img_shape = i1.shape
        height = img_shape[0]
        width = img_shape[1]

        # clamp bounds to valid range
        x_min = np.clip(x_min, -(width - 1), width - 1)
        x_max = np.clip(x_max, -(width - 1), width - 1)
        y_min = np.clip(y_min, -(height - 1), height - 1)
        y_max = np.clip(y_max, -(height - 1), height - 1)

        # create array of peaks +1 for inclusive
        cache = np.nan * np.ones((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

        peak_results = list()
        # evaluate the starting point hill climb
        peak = Refine.hill_climb_worker(i1, i2, x_min, x_max, y_min, y_max, start_x, start_y, cache, backend)
        peak_results.append(peak)

        # perform the random starting point multipoint hill climbing
        for i in range(num_hill_climbs - 1):
            # generate random starting point
            start_x = np.random.randint(x_min, x_max + 1)
            start_y = np.random.randint(y_min, y_max + 1)

            peak = Refine.hill_climb_worker(i1, i2, x_min, x_max, y_min, y_max, start_x, start_y, cache, backend)
            peak_results.append(peak)

        # find the best correlation and translation from the hill climb ending points
        best_index = np.argmax([peak.ncc for peak in peak_results])
        best_peak = peak_results[best_index]

        # determine how many converged
        converged = np.sum([1 for peak in peak_results if peak.x == best_peak.x and peak.y == best_peak.y])

        elapsed_time = time.time() - start_time
        backend_name = backend.__class__.__name__ if backend else "Legacy CPU"
        logging.info(f"Translation Hill Climb ({t1.name}, {t2.name}) using {backend_name} backend: "
                    f"{converged}/{num_hill_climbs} hill climbs converged with best ncc = {best_peak.ncc} "
                    f"in {elapsed_time:.3f} seconds")

        return best_peak

    @staticmethod
    def optimize_direction(tile: img_tile.Tile, other: img_tile.Tile, direction: str,
                          repeatability: int, num_hill_climbs: int,
                          backend: ComputeBackend = None) -> img_tile.Peak:
        """
        Optimizes the translation in a given direction using hill climbing.

        Args:
            tile: The tile to optimize
            other: The neighboring tile (north or west)
            direction: The direction to optimize ('west' or 'north')
            repeatability: The search radius for hill climbing
            num_hill_climbs: Number of hill climbs to perform
            backend: ComputeBackend instance to use for computation (optional)

        Returns:
            The optimized Peak
        """
        assert direction in ['west', 'north']
        relevant_translation = tile.west_translation if direction == 'west' else tile.north_translation
        orig_peak = copy.deepcopy(relevant_translation)
        x_min = orig_peak.x - repeatability
        x_max = orig_peak.x + repeatability
        y_min = orig_peak.y - repeatability
        y_max = orig_peak.y + repeatability

        new_peak = Refine.multipoint_hill_climb(num_hill_climbs, other, tile,
                                              x_min, x_max, y_min, y_max,
                                              orig_peak.x, orig_peak.y, backend)

        # If the old correlation was a number, then it was a good translation.
        # Increment the new translation by the value of the old correlation to increase beyond 1
        # This will enable these tiles to have higher priority in minimum spanning tree search
        if not np.isnan(orig_peak.ncc):
            new_peak.ncc += 3.0

        return new_peak


class RefineSequential(Refine):
    def __init__(self, args: argparse.Namespace, tile_grid: img_grid.TileGrid, sm: stage_model.StageModel):
        """
        Initialize the sequential refinement with the given arguments.

        Args:
            args: Command-line arguments
            tile_grid: The grid of tiles to refine
            sm: The stage model
        """
        self.args = args
        self.tile_grid = tile_grid
        self.sm = sm
        self.backend = create_compute_backend(args)
        logging.info(f"Initialized RefineSequential with backend: {self.backend.__class__.__name__}")

    def execute(self):
        """
        Execute the sequential refinement process.
        """
        logging.info("Starting Translation Refinement")
        start_time = time.time()
        # iterate over the tile grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = self.tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = self.tile_grid.get_tile(r, c - 1)
                if west is not None:
                    # optimize with west neighbor
                    tile.west_translation = Refine.optimize_direction(
                        tile, west, 'west', self.sm.repeatability,
                        self.args.num_hill_climbs, self.backend
                    )

                north = self.tile_grid.get_tile(r - 1, c)
                if north is not None:
                    # optimize with north neighbor
                    tile.north_translation = Refine.optimize_direction(
                        tile, north, 'north', self.sm.repeatability,
                        self.args.num_hill_climbs, self.backend
                    )

        elapsed_time = time.time() - start_time
        logging.info(f"Translation Refinement took {elapsed_time:.3f} seconds using {self.backend.__class__.__name__} backend")


class RefineParallel(Refine):
    def __init__(self, args: argparse.Namespace, tile_grid: img_grid.TileGrid, sm: stage_model.StageModel):
        """
        Initialize the parallel refinement with the given arguments.

        Args:
            args: Command-line arguments
            tile_grid: The grid of tiles to refine
            sm: The stage model
        """
        self.args = args
        self.tile_grid = tile_grid
        self.sm = sm
        # We don't create a backend here because each worker will create its own
        logging.info(f"Initialized RefineParallel (each worker will create its own backend)")

    @staticmethod
    def _worker_with_backend(tile: img_tile.Tile, other: img_tile.Tile, direction: str,
                           repeatability: int, num_hill_climbs: int, r: int, c: int,
                           args: argparse.Namespace) -> tuple[img_tile.Peak, int, int, str]:
        """
        Worker function that creates its own backend instance.

        Args:
            tile: The tile to optimize
            other: The neighboring tile
            direction: The direction to optimize ('west' or 'north')
            repeatability: The search radius for hill climbing
            num_hill_climbs: Number of hill climbs to perform
            r: Row index
            c: Column index
            args: Command-line arguments for creating the backend

        Returns:
            A tuple of (peak, row, column, direction)
        """
        # Create a new backend instance for this worker
        backend = create_compute_backend(args)

        # Optimize the direction using the backend
        peak = Refine.optimize_direction(tile, other, direction, repeatability, num_hill_climbs, backend)

        return peak, r, c, direction

    def execute(self):
        """
        Execute the parallel refinement process.
        """
        logging.info("Starting Translation Refinement in parallel")
        start_time = time.time()

        worker_input_list = list()
        # iterate over the tile grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = self.tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = self.tile_grid.get_tile(r, c - 1)
                if west is not None:
                    # optimize with west neighbor
                    worker_input_list.append((tile, west, 'west', self.sm.repeatability, self.args.num_hill_climbs, r, c, self.args))

                north = self.tile_grid.get_tile(r - 1, c)
                if north is not None:
                    # optimize with north neighbor
                    worker_input_list.append((tile, north, 'north', self.sm.repeatability, self.args.num_hill_climbs, r, c, self.args))

        import multiprocessing
        if hasattr(self.args, 'num_threads'):
            processes = self.args.num_threads
        else:
            processes = utils.get_num_workers()

        with multiprocessing.Pool(processes=processes) as pool:
            # perform the work in parallel
            results = pool.starmap(RefineParallel._worker_with_backend, worker_input_list)

        for result in results:
            peak, r, c, direction = result
            tile = self.tile_grid.get_tile(r, c)
            if tile is not None:
                if direction == 'west':
                    tile.west_translation = peak
                elif direction == 'north':
                    tile.north_translation = peak

        elapsed_time = time.time() - start_time
        logging.info(f"Translation Refinement took {elapsed_time:.3f} seconds using {processes} processes")


class GlobalPositions():
    """
    No changes needed for this class as it doesn't interact with the backend.
    """
    # Existing implementation remains unchanged
```
