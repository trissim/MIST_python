# plan_01c_gpu_hooks_pciam.md
## Component: PCIAM Modifications

### Objective
Modify the PCIAM class to use the ComputeBackend interface for its computational operations.

### Implementation Draft

#### 1. Modify PCIAM to Use the Backend Interface

We'll update the PCIAM class to use the backend interface:

```python
import argparse
import numpy as np
import scipy.fft
import time
import logging
from abc import ABC

# local imports
import MIST.img_tile as img_tile
import MIST.img_grid as img_grid
import MIST.utils as utils
from MIST.compute_backend import ComputeBackend
from MIST.backend_factory import create_compute_backend

class PCIAM(ABC):
    """
    Phase Correlation Image Alignment Method (PCIAM) base class.
    This class now delegates computational operations to a ComputeBackend instance.
    """
    
    @staticmethod
    def extract_subregion(t1: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.extract_subregion(t1, x, y)
    
    @staticmethod
    def cross_correlation(a1: np.ndarray, a2: np.ndarray) -> float:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.cross_correlation(a1, a2)
    
    @staticmethod
    def compute_cross_correlation(t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> float:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.compute_cross_correlation(t1, t2, x, y)
    
    @staticmethod
    def peak_cross_correlation_worker(t1: np.ndarray, t2: np.ndarray, dims: list[tuple[int, int]]) -> img_tile.Peak:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.peak_cross_correlation_worker(t1, t2, dims)
    
    @staticmethod
    def peak_cross_correlation_lr(t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.peak_cross_correlation_lr(t1, t2, x, y)
    
    @staticmethod
    def peak_cross_correlation_ud(t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.peak_cross_correlation_ud(t1, t2, x, y)
    
    @staticmethod
    def compute_pciam(t1: img_tile.Tile, t2: img_tile.Tile, n_peaks: int) -> img_tile.Peak:
        """
        Legacy static method that delegates to the CPU backend.
        Kept for backward compatibility.
        """
        from MIST.cpu_backend import CPUBackend
        cpu_backend = CPUBackend()
        return cpu_backend.compute_pciam(t1, t2, n_peaks)


class PciamSequential(PCIAM):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = create_compute_backend(args)
        logging.info(f"Initialized PciamSequential with backend: {self.backend.__class__.__name__}")

    def execute(self, tile_grid: img_grid.TileGrid):
        start_time = time.time()
        # iterate over the rows and columns of the grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = tile_grid.get_tile(r, c - 1)
                if west is not None:
                    peak = self.backend.compute_pciam(west, tile, self.args.num_fft_peaks)
                    tile.west_translation = peak

                north = tile_grid.get_tile(r - 1, c)
                if north is not None:
                    peak = self.backend.compute_pciam(north, tile, self.args.num_fft_peaks)
                    tile.north_translation = peak

        elapsed_time = time.time() - start_time
        logging.info("Finished computing all pairwise translations in {} seconds".format(elapsed_time))


class PciamParallel(PCIAM):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = create_compute_backend(args)
        logging.info(f"Initialized PciamParallel with backend: {self.backend.__class__.__name__}")

    def _worker(self, tile: img_tile.Tile, other: img_tile.Tile, r: int, c: int, direction: str, num_fft_peaks) -> tuple[img_tile.Peak, int, int, str]:
        # Note: In parallel mode, each worker gets its own backend instance
        # This is because the backend might not be thread-safe
        backend = create_compute_backend(self.args)
        return backend.compute_pciam(other, tile, num_fft_peaks), r, c, direction

    def execute(self, tile_grid: img_grid.TileGrid):
        start_time = time.time()
        logging.info("Computing all pairwise translations in parallel")
        logging.info("Preloading all images into memory")
        tile_grid.load_images_into_memory()
        logging.info("Finished preloading all images into memory. Took {}s".format(time.time() - start_time))

        worker_input_list = list()
        # iterate over the rows and columns of the grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = tile_grid.get_tile(r, c - 1)
                if west is not None:
                    worker_input_list.append((tile, west, r, c, 'west', self.args.num_fft_peaks))

                north = tile_grid.get_tile(r - 1, c)
                if north is not None:
                    worker_input_list.append((tile, north, r, c, 'north', self.args.num_fft_peaks))

        import multiprocessing
        if hasattr(self.args, 'num_threads'):
            processes = self.args.num_threads
        else:
            processes = utils.get_num_workers()
        with multiprocessing.Pool(processes=processes) as pool:
            # perform the work in parallel
            results = pool.starmap(self._worker, worker_input_list)

        for result in results:
            peak, r, c, direction = result
            tile = tile_grid.get_tile(r, c)
            if tile is not None:
                if direction == 'west':
                    tile.west_translation = peak
                elif direction == 'north':
                    tile.north_translation = peak

        elapsed_time = time.time() - start_time
        logging.info("Finished computing all pairwise translations in {} seconds".format(elapsed_time))
```
