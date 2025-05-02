# plan_01_gpu_hooks.md
## Component: GPU Backend Hooks

### Objective
Create a pluggable backend architecture for the MIST image processing pipeline that allows for GPU acceleration while maintaining the existing CPU functionality as the default. The architecture should enable toggling between CPU and GPU modes via a runtime flag or configuration option.

### Plan
1. Create an abstract backend interface for the core computation functions in PCIAM
   - Define a `ComputeBackend` abstract base class with methods for the key computational operations
   - Extract the core computation methods from PCIAM into this interface
   - Focus on `compute_cross_correlation` and related methods as the primary acceleration targets

2. Implement a CPU backend that maintains the current functionality
   - Create a `CPUBackend` class that implements the `ComputeBackend` interface
   - Move the existing implementation code from PCIAM to this class
   - Ensure all functionality remains identical to the current implementation

3. Modify PCIAM to use the backend interface
   - Add a backend selection mechanism to PCIAM
   - Update PCIAM to delegate computation to the selected backend
   - Ensure all existing code paths continue to work with the CPU backend

4. Add command-line arguments for GPU mode selection
   - Add a `--use-gpu` flag to the argument parsers in main.py and main_csv.py
   - Add a `--gpu-device` option to specify which GPU to use (if multiple are available)
   - Pass these options to the PCIAM constructor

5. Create a factory method for backend instantiation
   - Implement a factory function that creates the appropriate backend based on arguments
   - Handle fallback to CPU if GPU is requested but unavailable
   - Add appropriate logging for backend selection

6. Extend utils.py with GPU detection capabilities
   - Add functions to check for CUDA/GPU availability
   - Implement device enumeration for multi-GPU systems
   - Create helper functions for GPU memory management and device selection
   - Support graceful fallback to CPU when GPU is unavailable or insufficient

7. Create extension points for future GPU optimizations
   - Design the backend interface with hooks for batch processing of multiple tiles
   - Add placeholder methods for future GPU-accelerated hill climbing
   - Ensure the architecture can accommodate different optimization strategies (e.g., memory vs. speed tradeoffs)
   - Provide extension points for GPU-specific parameters and tuning options

### Findings
The MIST codebase is structured around a grid-based stitching pipeline that uses normalized cross-correlation (NCC) to compute image alignment. The key computational components are:

1. **PCIAM Class**: An abstract base class that defines the core computation methods:
   - `compute_cross_correlation`: Computes NCC between two image tiles with a given offset
   - `cross_correlation`: Computes NCC between two arrays
   - `extract_subregion`: Extracts a subregion from an image based on translation
   - `peak_cross_correlation_worker`: Computes NCC for multiple offsets
   - `compute_pciam`: Main entry point for computing phase correlation

2. **PciamSequential and PciamParallel**: Concrete implementations of PCIAM that provide sequential and parallel execution paths.

3. **Translation Refinement**: Uses hill climbing to refine tile translations, heavily relying on `compute_cross_correlation`.

The most GPU-relevant hotspot is `PCIAM.compute_cross_correlation`, which is called intensively in the hill climbing loop across large grids of tiles. This method and its dependencies (`cross_correlation` and `extract_subregion`) are the primary targets for GPU acceleration.

The codebase already has a pattern for selecting between sequential and parallel implementations based on a command-line flag (`--disable-mem-cache`), which can serve as a model for the CPU/GPU selection mechanism.

The utils.py module currently handles thread detection and environment checks, making it a natural place to extend with GPU detection capabilities. Additionally, both the NCC computation in PCIAM and the hill climbing algorithm in translation_refinement.py could benefit from deeper GPU integration in future iterations, so the architecture should be designed with these extension points in mind.

### Implementation Draft

#### 1. Create the ComputeBackend Abstract Base Class

First, we'll create a new file `compute_backend.py` that defines the abstract base class for the compute backends:

```python
import numpy as np
from abc import ABC, abstractmethod
import MIST.img_tile as img_tile

class ComputeBackend(ABC):
    """
    Abstract base class for compute backends that implement the core computational
    operations used in the MIST image processing pipeline.
    """

    @abstractmethod
    def extract_subregion(self, t1: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Extracts the sub-region visible if the image view window is translated the given (x,y) distance.

        Args:
            t1: The image tile a sub-region is being extracted from. The translation (x,y) is relative to the upper left corner of this image.
            x: The x component of the translation.
            y: The y component of the translation.

        Returns:
            The portion of tile shown if the view is translated (x,y) pixels, or None if there's no overlap.
        """
        pass

    @abstractmethod
    def cross_correlation(self, a1: np.ndarray, a2: np.ndarray) -> float:
        """
        Computes the cross correlation between two arrays.

        Args:
            a1: The first array
            a2: The second array

        Returns:
            The normalized cross correlation between the two arrays.
        """
        pass

    @abstractmethod
    def compute_cross_correlation(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> float:
        """
        Computes the cross correlation between two ImageTiles given the offset (x,y) from the first to the second.

        Args:
            t1: The first tile
            t2: The second tile
            x: The x component of the translation from t1 to t2.
            y: The y component of the translation from t1 to t2.

        Returns:
            The normalized cross correlation between the overlapping pixels given the translation between t1 and t2 (x,y).
        """
        pass

    @abstractmethod
    def peak_cross_correlation_worker(self, t1: np.ndarray, t2: np.ndarray, dims: list[tuple[int, int]]) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images for multiple offset dimensions.

        Args:
            t1: The first image
            t2: The second image
            dims: List of (y, x) offset tuples to check

        Returns:
            The Peak with the highest NCC value.
        """
        pass

    @abstractmethod
    def peak_cross_correlation_lr(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the left image.

        Args:
            t1: The left image
            t2: The right image
            x: The x component of the initial offset
            y: The y component of the initial offset

        Returns:
            The Peak with the highest NCC value.
        """
        pass

    @abstractmethod
    def peak_cross_correlation_ud(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the top image.

        Args:
            t1: The top image
            t2: The bottom image
            x: The x component of the initial offset
            y: The y component of the initial offset

        Returns:
            The Peak with the highest NCC value.
        """
        pass

    @abstractmethod
    def compute_pciam(self, t1: img_tile.Tile, t2: img_tile.Tile, n_peaks: int) -> img_tile.Peak:
        """
        Computes the phase correlation between two image tiles and returns the peak with highest NCC.

        Args:
            t1: The first tile
            t2: The second tile
            n_peaks: Number of peaks to consider from the phase correlation

        Returns:
            The Peak with the highest NCC value.
        """
        pass

    # Extension points for future GPU optimizations

    @abstractmethod
    def batch_compute_cross_correlation(self, t1: np.ndarray, t2: np.ndarray, offsets: list[tuple[int, int]]) -> list[float]:
        """
        Computes cross correlation for multiple offsets in a batch operation.
        This is a placeholder for future GPU optimization.

        Args:
            t1: The first image
            t2: The second image
            offsets: List of (x, y) offset tuples

        Returns:
            List of NCC values corresponding to each offset
        """
        pass

    @abstractmethod
    def hill_climb(self, t1: np.ndarray, t2: np.ndarray, initial_x: int, initial_y: int,
                  search_radius: int, cache: np.ndarray = None) -> img_tile.Peak:
        """
        Performs hill climbing to find the peak correlation.
        This is a placeholder for future GPU-accelerated hill climbing.

        Args:
            t1: The first image
            t2: The second image
            initial_x: Initial x offset
            initial_y: Initial y offset
            search_radius: Radius to search around the initial point
            cache: Optional cache of previously computed values

        Returns:
            The Peak with the highest NCC value found by hill climbing
        """
        pass
```

The implementation is split across multiple plan files for better organization:

1. **plan_01a_gpu_hooks_cpu_backend.md**: Contains the CPU backend implementation that wraps the existing PCIAM functionality.

2. **plan_01b_gpu_hooks_utils.md**: Contains the GPU detection utilities to be added to utils.py and the backend factory function.

3. **plan_01c_gpu_hooks_pciam.md**: Contains the modifications to the PCIAM class to use the ComputeBackend interface.

4. **plan_01d_gpu_hooks_cli.md**: Contains the command-line argument additions and the backend factory module.
