# plan_01a_gpu_hooks_cpu_backend.md
## Component: CPU Backend Implementation

### Objective
Implement the CPU backend that wraps the existing PCIAM functionality to conform to the new ComputeBackend interface.

### Implementation Draft

#### 1. Implement the CPU Backend

We'll create a CPU backend that implements the ComputeBackend interface using the existing CPU-based implementation:

```python
import numpy as np
import scipy.fft
import MIST.img_tile as img_tile
from MIST.compute_backend import ComputeBackend

class CPUBackend(ComputeBackend):
    """
    CPU implementation of the ComputeBackend interface.
    This class wraps the existing CPU-based implementation from PCIAM.
    """
    
    def extract_subregion(self, t1: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Extracts the sub-region visible if the image view window is translated the given (x,y) distance.
        """
        w = t1.shape[1]
        h = t1.shape[0]

        x_start = x
        x_end = x + w - 1
        y_start = y
        y_end = y + h - 1

        # constrain to valid coordinates
        x_start = np.clip(x_start, 0, w - 1)
        x_end = np.clip(x_end, 0, w - 1)
        y_start = np.clip(y_start, 0, h - 1)
        y_end = np.clip(y_end, 0, h - 1)

        # if the translations (x,y) would leave no overlap between the images, return None
        if abs(x) >= w or abs(y) >= h:
            return None

        sub_tile = t1[y_start:y_end + 1, x_start:x_end + 1]
        return sub_tile
    
    def cross_correlation(self, a1: np.ndarray, a2: np.ndarray) -> float:
        """
        Computes the cross correlation between two arrays.
        """
        a1 = a1.ravel().astype(np.float32)
        a2 = a2.ravel().astype(np.float32)

        a1 -= np.mean(a1)
        a2 -= np.mean(a2)

        neumerator = np.matmul(a1.transpose(), a2)
        denominator = np.sqrt(np.matmul(a1.transpose(), a1) * np.matmul(a2.transpose(), a2))
        cr = neumerator / denominator
        if not np.isfinite(cr):
            cr = -1

        return cr
    
    def compute_cross_correlation(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> float:
        """
        Computes the cross correlation between two ImageTiles given the offset (x,y) from the first to the second.
        """
        a1 = self.extract_subregion(t1, x, y)
        if a1 is None:
            return -1.0
        a2 = self.extract_subregion(t2, -x, -y)
        if a2 is None:
            return -1.0
        return self.cross_correlation(a1, a2)
    
    def peak_cross_correlation_worker(self, t1: np.ndarray, t2: np.ndarray, dims: list[tuple[int, int]]) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images for multiple offset dimensions.
        """
        # remove duplicate dim values to prevent redundant computation
        dims = list(set(dims))

        ncc_list = list()
        x_list = list()
        y_list = list()
        for i in range(len(dims)):
            nr = dims[i][0]
            nc = dims[i][1]

            peak = self.compute_cross_correlation(t1, t2, nc, nr)
            if np.isnan(peak):
                peak = -1
            ncc_list.append(peak)
            x_list.append(nc)
            y_list.append(nr)

        idx = np.argmax(ncc_list)
        peak = img_tile.Peak(ncc_list[idx], x_list[idx], y_list[idx])

        return peak
    
    def peak_cross_correlation_lr(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the left image.
        """
        w = t1.shape[1]
        h = t1.shape[0]
        
        dims = [(y, x), (y, w - x), (h - y, x), (h - y, w - x),
                ((-y), x), ((-y), w - x), (-(h - y), x), (-(h - y), w - x)]

        return self.peak_cross_correlation_worker(t1, t2, dims)
    
    def peak_cross_correlation_ud(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the top image.
        """
        w = t1.shape[1]
        h = t1.shape[0]
        
        dims = [(y, x), (y, w - x), (h - y, x), (h - y, w - x),
                (y, (-x)), (y, -(w - x)), (h - y, (-x)), (h - y, -(w - x))]

        return self.peak_cross_correlation_worker(t1, t2, dims)
    
    def compute_pciam(self, t1: img_tile.Tile, t2: img_tile.Tile, n_peaks: int) -> img_tile.Peak:
        """
        Computes the phase correlation between two image tiles and returns the peak with highest NCC.
        """
        t1_img = t1.get_image()
        t2_img = t2.get_image()

        # use scipy over np to do fft in 32bit (for complex64 result)
        fc = scipy.fft.fft2(t1_img.astype(np.float32)) * np.conj(scipy.fft.fft2(t2_img.astype(np.float32)))
        np.clip(fc.real, a_min=1e-16, a_max=None, out=fc.real)  # specify out for in place clip
        np.clip(fc.imag, a_min=1e-16, a_max=None, out=fc.imag)  # specify out for in place clip
        fc = np.nan_to_num(fc, nan=1e-16, copy=False)  # replace nans with min value, copy=False for in place
        fcn = fc / np.abs(fc)
        pcm = np.real(scipy.fft.ifft2(fcn))

        # get the n_peaks largest values using argpartition to avoid sort
        indices = pcm.argpartition(pcm.size - n_peaks, axis=None)[-n_peaks:]

        peak_list = list()

        for ind in indices:
            y, x = np.unravel_index(ind, pcm.shape)
            if t1.r == t2.r:
                # same row, so compute NCC along Left-Right
                peak = self.peak_cross_correlation_lr(t1_img, t2_img, x, y)
            else:
                # different row, so compute NCC along Up-Down
                peak = self.peak_cross_correlation_ud(t1_img, t2_img, x, y)
            peak_list.append(peak)

        peak = max(peak_list, key=lambda p: p.ncc)
        return peak
    
    # Implementation of extension points (basic CPU versions)
    
    def batch_compute_cross_correlation(self, t1: np.ndarray, t2: np.ndarray, offsets: list[tuple[int, int]]) -> list[float]:
        """
        Basic CPU implementation of batch cross correlation computation.
        Simply loops through the offsets and computes each one individually.
        """
        results = []
        for x, y in offsets:
            ncc = self.compute_cross_correlation(t1, t2, x, y)
            results.append(ncc)
        return results
    
    def hill_climb(self, t1: np.ndarray, t2: np.ndarray, initial_x: int, initial_y: int, 
                  search_radius: int, cache: np.ndarray = None) -> img_tile.Peak:
        """
        Basic CPU implementation of hill climbing.
        This is a placeholder that just returns the NCC at the initial point.
        A full implementation would be added in a future plan.
        """
        ncc = self.compute_cross_correlation(t1, t2, initial_x, initial_y)
        return img_tile.Peak(ncc, initial_x, initial_y)
```
