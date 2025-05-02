import numpy as np
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import img_tile

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
