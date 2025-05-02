import numpy as np
import logging
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import img_tile
from MIST.compute_backend import ComputeBackend
import MIST.utils as utils

class GPUBackend(ComputeBackend):
    """
    GPU implementation of the ComputeBackend interface.
    This class provides GPU-accelerated versions of the core computational operations
    using either PyTorch or CuPy for CUDA acceleration.
    """

    def __init__(self, device_id=0):
        """
        Initialize the GPU backend with the specified device.
        Attempts to use PyTorch first, then falls back to CuPy if PyTorch is not available.

        Args:
            device_id (int): The ID of the GPU device to use

        Raises:
            RuntimeError: If GPU support is not available or initialization fails
        """
        self.device_id = device_id
        self.gpu_lib = None
        self.device = None
        self.stream = None
        self.memory_allocated = 0
        self.memory_reserved = 0
        self.max_memory_allocated = 0

        # Check if CUDA is available through utils
        if not utils.is_cuda_available():
            raise RuntimeError("GPUBackend requires CUDA, but it is not available on this system. "
                              "Please ensure CUDA is installed and compatible with your GPU.")

        # Check if the requested device is valid
        gpu_count = utils.get_gpu_count()
        if device_id >= gpu_count:
            raise RuntimeError(f"Requested GPU device {device_id} is not available. "
                              f"Only {gpu_count} devices are available.")

        # Try to initialize PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_lib = "torch"
                torch.cuda.set_device(device_id)
                self.device = torch.device(f"cuda:{device_id}")

                # Get device info
                device_props = torch.cuda.get_device_properties(device_id)
                logging.info(f"Initialized GPUBackend with PyTorch using device {device_id}: {device_props.name}")
                logging.info(f"CUDA Capability: {device_props.major}.{device_props.minor}")
                logging.info(f"Total GPU memory: {device_props.total_memory / (1024**3):.2f} GB")
                logging.info(f"CUDA version: {torch.version.cuda}")

                # Set up memory tracking
                self._update_memory_stats()
                logging.info(f"Initial GPU memory allocated: {self.memory_allocated / (1024**2):.2f} MB")
                logging.info(f"Initial GPU memory reserved: {self.memory_reserved / (1024**2):.2f} MB")

                # Enable tensor cores if available (for Volta+ GPUs)
                if hasattr(torch.backends.cudnn, 'allow_tf32') and device_props.major >= 7:
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    logging.info("Enabled TF32 precision for faster computation")

                # Set optimal algorithm selection for cudnn
                torch.backends.cudnn.benchmark = True

                return
        except (ImportError, Exception) as e:
            logging.warning(f"Failed to initialize PyTorch GPU backend: {str(e)}. Trying CuPy...")

        # Fall back to CuPy if PyTorch is not available or fails
        try:
            import cupy as cp
            self.gpu_lib = "cupy"
            self.cp = cp
            self.device = cp.cuda.Device(device_id)
            self.stream = cp.cuda.Stream()
            self.device.use()

            # Get device info
            device_info = utils.get_gpu_info(device_id)
            if device_info:
                logging.info(f"Initialized GPUBackend with CuPy using device {device_id}: {device_info['name']}")
                logging.info(f"GPU memory: {device_info['memory_available'] / (1024**2):.2f} MB available / "
                            f"{device_info['memory_total'] / (1024**2):.2f} MB total")

            # Set up memory pool for better performance
            with self.stream:
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(fraction=0.8)  # Use up to 80% of available memory
                logging.info(f"CuPy memory pool limit set to 80% of available memory")

            return
        except ImportError:
            raise RuntimeError("GPUBackend requires either PyTorch or CuPy, but neither could be imported. "
                              "Please install PyTorch with 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118' "
                              "or CuPy with 'pip install cupy-cuda11x' (replace with your CUDA version).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU backend: {str(e)}")

    def _update_memory_stats(self):
        """
        Update memory statistics for monitoring.
        """
        if self.gpu_lib == "torch":
            import torch
            self.memory_allocated = torch.cuda.memory_allocated(self.device_id)
            self.memory_reserved = torch.cuda.memory_reserved(self.device_id)
            self.max_memory_allocated = max(self.max_memory_allocated, self.memory_allocated)
        elif self.gpu_lib == "cupy":
            mempool = self.cp.get_default_memory_pool()
            self.memory_allocated = mempool.used_bytes()
            self.memory_reserved = mempool.total_bytes()
            self.max_memory_allocated = max(self.max_memory_allocated, self.memory_allocated)

    def _to_gpu(self, array):
        """
        Transfer a numpy array to the GPU.
        Handles different GPU libraries and includes error checking.

        Args:
            array: The numpy array to transfer

        Returns:
            The array on the GPU

        Raises:
            RuntimeError: If the transfer fails
        """
        if array is None:
            return None

        try:
            # Track memory before allocation
            before_mem = self.memory_allocated

            # Transfer data based on the GPU library being used
            if self.gpu_lib == "torch":
                import torch
                result = torch.from_numpy(array).to(self.device)

                # Force synchronization to ensure transfer is complete
                torch.cuda.synchronize(self.device)
            elif self.gpu_lib == "cupy":
                with self.stream:
                    result = self.cp.asarray(array)
                    # Force synchronization to ensure transfer is complete
                    self.stream.synchronize()
            else:
                raise RuntimeError(f"Unknown GPU library: {self.gpu_lib}")

            # Update memory stats and log if a large allocation occurred
            self._update_memory_stats()
            allocation_size = self.memory_allocated - before_mem
            if allocation_size > 100 * 1024 * 1024:  # Log allocations larger than 100MB
                logging.debug(f"Large GPU memory allocation: {allocation_size / (1024**2):.2f} MB, "
                             f"total: {self.memory_allocated / (1024**2):.2f} MB")

            return result

        except Exception as e:
            logging.error(f"Failed to transfer array to GPU: {str(e)}")
            logging.error(f"Array shape: {array.shape}, dtype: {array.dtype}")
            logging.error(f"Current GPU memory: {self.memory_allocated / (1024**2):.2f} MB / "
                         f"{self.memory_reserved / (1024**2):.2f} MB")

            # Try to recover by clearing some memory
            self._try_clear_memory()

            # Raise a more informative error
            raise RuntimeError(f"GPU transfer failed: {str(e)}. Consider using smaller images or a CPU backend.")

    def _to_cpu(self, array):
        """
        Transfer an array from the GPU to the CPU.
        Handles different GPU libraries and includes error checking.

        Args:
            array: The GPU array to transfer

        Returns:
            The array as a numpy array on the CPU

        Raises:
            RuntimeError: If the transfer fails
        """
        if array is None:
            return None

        try:
            # Transfer data based on the GPU library being used
            if self.gpu_lib == "torch":
                import torch
                # Ensure the operation is complete before returning
                torch.cuda.synchronize(self.device)
                return array.cpu().numpy()
            elif self.gpu_lib == "cupy":
                with self.stream:
                    result = self.cp.asnumpy(array)
                    # Ensure the operation is complete before returning
                    self.stream.synchronize()
                    return result
            else:
                raise RuntimeError(f"Unknown GPU library: {self.gpu_lib}")

        except Exception as e:
            logging.error(f"Failed to transfer array from GPU to CPU: {str(e)}")
            if hasattr(array, 'shape') and hasattr(array, 'dtype'):
                logging.error(f"Array shape: {array.shape}, dtype: {array.dtype}")
            logging.error(f"Current GPU memory: {self.memory_allocated / (1024**2):.2f} MB / "
                         f"{self.memory_reserved / (1024**2):.2f} MB")

            # Raise a more informative error
            raise RuntimeError(f"GPU to CPU transfer failed: {str(e)}.")

    def _try_clear_memory(self):
        """
        Attempt to clear GPU memory in case of memory pressure.
        """
        try:
            if self.gpu_lib == "torch":
                import torch
                # Empty cache to free up memory
                torch.cuda.empty_cache()
                logging.info("Cleared PyTorch CUDA cache to free memory")
            elif self.gpu_lib == "cupy":
                # Clear memory pool
                mempool = self.cp.get_default_memory_pool()
                mempool.free_all_blocks()
                logging.info("Cleared CuPy memory pool to free memory")

            # Update memory stats after clearing
            self._update_memory_stats()
            logging.info(f"After clearing: GPU memory allocated: {self.memory_allocated / (1024**2):.2f} MB, "
                        f"reserved: {self.memory_reserved / (1024**2):.2f} MB")
        except Exception as e:
            logging.error(f"Failed to clear GPU memory: {str(e)}")

    def extract_subregion(self, t1: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Extracts the sub-region visible if the image view window is translated the given (x,y) distance.
        This operation is performed on the CPU as it's not computationally intensive.

        Args:
            t1: The image tile a sub-region is being extracted from
            x: The x component of the translation
            y: The y component of the translation

        Returns:
            The extracted sub-region, or None if there's no overlap
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
        Computes the cross correlation between two arrays using GPU acceleration.
        Optimized for both PyTorch and CuPy backends with robust error handling.

        Args:
            a1: The first array
            a2: The second array

        Returns:
            The normalized cross correlation between the two arrays

        Raises:
            RuntimeError: If the computation fails and cannot be recovered
        """
        start_time = time.time()

        # Validate input arrays
        if a1.size != a2.size:
            logging.warning(f"Array size mismatch in cross_correlation: {a1.shape} vs {a2.shape}")
            return -1.0

        if a1.size == 0 or a2.size == 0:
            logging.warning("Empty array in cross_correlation")
            return -1.0

        try:
            # Ensure arrays are float32 for better GPU performance
            a1_flat = a1.ravel().astype(np.float32)
            a2_flat = a2.ravel().astype(np.float32)

            if self.gpu_lib == "torch":
                import torch

                # Transfer data to GPU
                g_a1 = self._to_gpu(a1_flat)
                g_a2 = self._to_gpu(a2_flat)

                # Compute mean and center the data
                a1_mean = torch.mean(g_a1)
                a2_mean = torch.mean(g_a2)
                g_a1 = g_a1 - a1_mean
                g_a2 = g_a2 - a2_mean

                # Check for zero variance
                a1_var = torch.sum(g_a1 * g_a1)
                a2_var = torch.sum(g_a2 * g_a2)

                if a1_var <= 1e-10 or a2_var <= 1e-10:
                    logging.debug("Near-zero variance in cross_correlation")
                    return -1.0

                # Compute cross correlation using dot product
                numerator = torch.dot(g_a1, g_a2)
                denominator = torch.sqrt(a1_var * a2_var)

                # Compute result and transfer back to CPU
                cr = (numerator / denominator).item()

            elif self.gpu_lib == "cupy":
                # Transfer data to GPU
                with self.stream:
                    g_a1 = self._to_gpu(a1_flat)
                    g_a2 = self._to_gpu(a2_flat)

                    # Compute mean and center the data
                    a1_mean = self.cp.mean(g_a1)
                    a2_mean = self.cp.mean(g_a2)
                    g_a1 = g_a1 - a1_mean
                    g_a2 = g_a2 - a2_mean

                    # Check for zero variance
                    a1_var = self.cp.sum(g_a1 * g_a1)
                    a2_var = self.cp.sum(g_a2 * g_a2)

                    if a1_var <= 1e-10 or a2_var <= 1e-10:
                        logging.debug("Near-zero variance in cross_correlation")
                        return -1.0

                    # Compute cross correlation using dot product
                    numerator = self.cp.dot(g_a1, g_a2)
                    denominator = self.cp.sqrt(a1_var * a2_var)

                    # Compute result and transfer back to CPU
                    cr = float(self._to_cpu(numerator / denominator))
            else:
                # Fallback to CPU implementation
                logging.warning("No GPU library available, falling back to CPU implementation")
                a1_flat -= np.mean(a1_flat)
                a2_flat -= np.mean(a2_flat)

                a1_var = np.sum(a1_flat * a1_flat)
                a2_var = np.sum(a2_flat * a2_flat)

                if a1_var <= 1e-10 or a2_var <= 1e-10:
                    return -1.0

                numerator = np.dot(a1_flat, a2_flat)
                denominator = np.sqrt(a1_var * a2_var)
                cr = float(numerator / denominator)

            # Handle invalid results
            if not np.isfinite(cr):
                cr = -1.0

            # Clamp to valid range [-1, 1]
            cr = max(-1.0, min(1.0, cr))

            elapsed_time = time.time() - start_time
            if elapsed_time > 0.01:  # Only log if it took more than 10ms
                logging.debug(f"GPU cross_correlation took {elapsed_time:.6f} seconds for arrays of size {a1.size}")

            return cr

        except Exception as e:
            logging.warning(f"GPU cross_correlation failed: {str(e)}. Falling back to CPU implementation.")

            try:
                # Fallback to CPU implementation
                a1_flat = a1.ravel().astype(np.float32)
                a2_flat = a2.ravel().astype(np.float32)

                a1_flat -= np.mean(a1_flat)
                a2_flat -= np.mean(a2_flat)

                a1_var = np.sum(a1_flat * a1_flat)
                a2_var = np.sum(a2_flat * a2_flat)

                if a1_var <= 1e-10 or a2_var <= 1e-10:
                    return -1.0

                numerator = np.dot(a1_flat, a2_flat)
                denominator = np.sqrt(a1_var * a2_var)
                cr = float(numerator / denominator)

                if not np.isfinite(cr):
                    cr = -1.0

                # Clamp to valid range [-1, 1]
                cr = max(-1.0, min(1.0, cr))

                return cr

            except Exception as e2:
                logging.error(f"Both GPU and CPU cross_correlation failed: {str(e2)}")
                return -1.0

    def compute_cross_correlation(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> float:
        """
        Computes the cross correlation between two ImageTiles given the offset (x,y) from the first to the second.

        Args:
            t1: The first tile
            t2: The second tile
            x: The x component of the translation from t1 to t2
            y: The y component of the translation from t1 to t2

        Returns:
            The normalized cross correlation between the overlapping pixels
        """
        # Extract subregions on CPU (more efficient for small operations)
        a1 = self.extract_subregion(t1, x, y)
        if a1 is None:
            return -1.0
        a2 = self.extract_subregion(t2, -x, -y)
        if a2 is None:
            return -1.0

        # Compute cross correlation on GPU
        return self.cross_correlation(a1, a2)

    def peak_cross_correlation_worker(self, t1: np.ndarray, t2: np.ndarray, dims: list[tuple[int, int]]) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images for multiple offset dimensions.

        Args:
            t1: The first image
            t2: The second image
            dims: List of (y, x) offset tuples to check

        Returns:
            The Peak with the highest NCC value
        """
        # Remove duplicate dim values to prevent redundant computation
        dims = list(set(dims))

        # Use batch computation for better GPU utilization
        # Note: dims are (y, x) but offsets need to be (x, y)
        offsets = [(nc, nr) for nr, nc in dims]
        ncc_list = self.batch_compute_cross_correlation(t1, t2, offsets)

        # Create lists of x, y coordinates and NCC values
        x_list = [nc for _, nc in dims]
        y_list = [nr for nr, _ in dims]

        # Find the peak with the highest NCC
        idx = np.argmax(ncc_list)
        peak = img_tile.Peak(ncc_list[idx], x_list[idx], y_list[idx])

        return peak

    def peak_cross_correlation_lr(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the left image.

        Args:
            t1: The left image
            t2: The right image
            x: The x component of the initial offset
            y: The y component of the initial offset

        Returns:
            The Peak with the highest NCC value
        """
        w = t1.shape[1]
        h = t1.shape[0]

        dims = [(y, x), (y, w - x), (h - y, x), (h - y, w - x),
                ((-y), x), ((-y), w - x), (-(h - y), x), (-(h - y), w - x)]

        return self.peak_cross_correlation_worker(t1, t2, dims)

    def peak_cross_correlation_ud(self, t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the top image.

        Args:
            t1: The top image
            t2: The bottom image
            x: The x component of the initial offset
            y: The y component of the initial offset

        Returns:
            The Peak with the highest NCC value
        """
        w = t1.shape[1]
        h = t1.shape[0]

        dims = [(y, x), (y, w - x), (h - y, x), (h - y, w - x),
                (y, (-x)), (y, -(w - x)), (h - y, (-x)), (h - y, -(w - x))]

        return self.peak_cross_correlation_worker(t1, t2, dims)

    def compute_pciam(self, t1: img_tile.Tile, t2: img_tile.Tile, n_peaks: int) -> img_tile.Peak:
        """
        Computes the phase correlation between two image tiles and returns the peak with highest NCC.

        Args:
            t1: The first tile
            t2: The second tile
            n_peaks: Number of peaks to consider from the phase correlation

        Returns:
            The Peak with the highest NCC value
        """
        start_time = time.time()

        try:
            # Get images
            t1_img = t1.get_image()
            t2_img = t2.get_image()

            if self.gpu_lib == "torch":
                import torch

                # Transfer to GPU
                g_t1_img = self._to_gpu(t1_img.astype(np.float32))
                g_t2_img = self._to_gpu(t2_img.astype(np.float32))

                # Compute FFT
                g_t1_fft = torch.fft.fft2(g_t1_img)
                g_t2_fft = torch.fft.fft2(g_t2_img)

                # Compute phase correlation
                g_fc = g_t1_fft * torch.conj(g_t2_fft)

                # Clip and normalize
                g_fc_real = torch.clamp(g_fc.real, min=1e-16)
                g_fc_imag = torch.clamp(g_fc.imag, min=1e-16)
                g_fc = torch.complex(g_fc_real, g_fc_imag)
                g_fc = torch.nan_to_num(g_fc, nan=1e-16)
                g_fcn = g_fc / torch.abs(g_fc)
                g_pcm = torch.fft.ifft2(g_fcn).real

                # Transfer PCM back to CPU for peak finding
                pcm = g_pcm.cpu().numpy()

            elif self.gpu_lib == "cupy":
                # Transfer to GPU
                with self.stream:
                    g_t1_img = self._to_gpu(t1_img.astype(np.float32))
                    g_t2_img = self._to_gpu(t2_img.astype(np.float32))

                    # Compute FFT
                    g_t1_fft = self.cp.fft.fft2(g_t1_img)
                    g_t2_fft = self.cp.fft.fft2(g_t2_img)

                    # Compute phase correlation
                    g_fc = g_t1_fft * self.cp.conj(g_t2_fft)

                    # Clip and normalize
                    self.cp.clip(g_fc.real, a_min=1e-16, a_max=None, out=g_fc.real)
                    self.cp.clip(g_fc.imag, a_min=1e-16, a_max=None, out=g_fc.imag)
                    g_fc = self.cp.nan_to_num(g_fc, nan=1e-16)
                    g_fcn = g_fc / self.cp.abs(g_fc)
                    g_pcm = self.cp.real(self.cp.fft.ifft2(g_fcn))

                    # Transfer PCM back to CPU for peak finding
                    pcm = self._to_cpu(g_pcm)
            else:
                # Fallback to CPU implementation
                logging.warning("No GPU library available, falling back to CPU implementation for PCIAM")

                # Use scipy over np to do fft in 32bit (for complex64 result)
                import scipy.fft
                fc = scipy.fft.fft2(t1_img.astype(np.float32)) * np.conj(scipy.fft.fft2(t2_img.astype(np.float32)))
                np.clip(fc.real, a_min=1e-16, a_max=None, out=fc.real)
                np.clip(fc.imag, a_min=1e-16, a_max=None, out=fc.imag)
                fc = np.nan_to_num(fc, nan=1e-16, copy=False)
                fcn = fc / np.abs(fc)
                pcm = np.real(scipy.fft.ifft2(fcn))

            # Get the n_peaks largest values
            indices = pcm.argpartition(pcm.size - n_peaks, axis=None)[-n_peaks:]

            # Process each peak
            peak_list = []
            for ind in indices:
                y, x = np.unravel_index(ind, pcm.shape)
                if t1.r == t2.r:
                    # Same row, compute NCC along Left-Right
                    peak = self.peak_cross_correlation_lr(t1_img, t2_img, x, y)
                else:
                    # Different row, compute NCC along Up-Down
                    peak = self.peak_cross_correlation_ud(t1_img, t2_img, x, y)
                peak_list.append(peak)

            # Find the peak with the highest NCC
            peak = max(peak_list, key=lambda p: p.ncc)

            elapsed_time = time.time() - start_time
            logging.debug(f"GPU compute_pciam took {elapsed_time:.6f} seconds")

            return peak

        except Exception as e:
            logging.warning(f"GPU compute_pciam failed: {str(e)}. Falling back to CPU implementation.")

            try:
                # Try to recover by clearing memory
                self._try_clear_memory()

                # Fallback to CPU implementation
                from MIST.cpu_backend import CPUBackend
                cpu_backend = CPUBackend()
                return cpu_backend.compute_pciam(t1, t2, n_peaks)

            except Exception as e2:
                logging.error(f"Both GPU and CPU compute_pciam failed: {str(e2)}")

                # Return a default peak as last resort
                return img_tile.Peak(ncc=-1.0, x=0, y=0)

    def batch_compute_cross_correlation(self, t1: np.ndarray, t2: np.ndarray, offsets: list[tuple[int, int]]) -> list[float]:
        """
        Computes cross correlation for multiple offsets in a batch operation.
        Optimized for both PyTorch and CuPy backends with robust error handling and memory management.

        Args:
            t1: The first image
            t2: The second image
            offsets: List of (x, y) offset tuples

        Returns:
            List of NCC values corresponding to each offset
        """
        start_time = time.time()

        # Validate inputs
        if not offsets:
            return []

        # Pre-allocate results array
        results = [-1.0] * len(offsets)

        try:
            # Determine optimal batch size based on available GPU memory and image size
            # For larger images, use smaller batches to avoid OOM errors
            img_size_mb = t1.size * t1.itemsize / (1024 * 1024)
            adaptive_batch_size = min(64, max(1, int(1024 / (img_size_mb + 1))))
            batch_size = min(len(offsets), adaptive_batch_size)

            logging.debug(f"Using batch size {batch_size} for images of size {t1.shape}")

            # Process in batches to avoid GPU memory issues
            for batch_start in range(0, len(offsets), batch_size):
                batch_end = min(batch_start + batch_size, len(offsets))
                batch_offsets = offsets[batch_start:batch_end]

                # Extract subregions for each offset
                subregions1 = []
                subregions2 = []
                valid_indices = []

                for idx, (x, y) in enumerate(batch_offsets):
                    a1 = self.extract_subregion(t1, x, y)
                    a2 = self.extract_subregion(t2, -x, -y)

                    if a1 is not None and a2 is not None and a1.size > 0 and a2.size > 0 and a1.size == a2.size:
                        subregions1.append(a1.ravel().astype(np.float32))
                        subregions2.append(a2.ravel().astype(np.float32))
                        valid_indices.append(batch_start + idx)

                if not subregions1:
                    # No valid subregions in this batch
                    continue

                # Process the valid subregions based on the GPU library
                if self.gpu_lib == "torch":
                    import torch

                    # Convert to GPU tensors
                    g_subregions1 = self._to_gpu(np.array(subregions1))
                    g_subregions2 = self._to_gpu(np.array(subregions2))

                    # Center the data
                    means1 = torch.mean(g_subregions1, dim=1, keepdim=True)
                    means2 = torch.mean(g_subregions2, dim=1, keepdim=True)
                    g_subregions1 = g_subregions1 - means1
                    g_subregions2 = g_subregions2 - means2

                    # Compute variances
                    var1 = torch.sum(g_subregions1 * g_subregions1, dim=1)
                    var2 = torch.sum(g_subregions2 * g_subregions2, dim=1)

                    # Check for zero variance
                    valid_mask = (var1 > 1e-10) & (var2 > 1e-10)

                    if torch.any(valid_mask):
                        # Compute cross correlation for valid subregions
                        numerator = torch.sum(g_subregions1 * g_subregions2, dim=1)
                        denominator = torch.sqrt(var1 * var2)

                        # Compute correlation and handle invalid results
                        correlation = torch.zeros_like(numerator)
                        correlation[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

                        # Transfer to CPU and handle invalid values
                        correlation_cpu = correlation.cpu().numpy()
                        correlation_cpu[~np.isfinite(correlation_cpu)] = -1.0

                        # Update results with computed values
                        for i, idx in enumerate(valid_indices):
                            if valid_mask[i].item():
                                results[idx] = float(correlation_cpu[i])

                elif self.gpu_lib == "cupy":
                    # Convert to GPU arrays
                    with self.stream:
                        g_subregions1 = self._to_gpu(np.array(subregions1))
                        g_subregions2 = self._to_gpu(np.array(subregions2))

                        # Center the data
                        means1 = self.cp.mean(g_subregions1, axis=1, keepdims=True)
                        means2 = self.cp.mean(g_subregions2, axis=1, keepdims=True)
                        g_subregions1 = g_subregions1 - means1
                        g_subregions2 = g_subregions2 - means2

                        # Compute variances
                        var1 = self.cp.sum(g_subregions1 * g_subregions1, axis=1)
                        var2 = self.cp.sum(g_subregions2 * g_subregions2, axis=1)

                        # Check for zero variance
                        valid_mask = (var1 > 1e-10) & (var2 > 1e-10)

                        if self.cp.any(valid_mask):
                            # Compute cross correlation for valid subregions
                            numerator = self.cp.sum(g_subregions1 * g_subregions2, axis=1)
                            denominator = self.cp.sqrt(var1 * var2)

                            # Compute correlation and handle invalid results
                            correlation = self.cp.zeros_like(numerator)
                            correlation[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

                            # Transfer to CPU and handle invalid values
                            correlation_cpu = self._to_cpu(correlation)
                            correlation_cpu[~np.isfinite(correlation_cpu)] = -1.0

                            # Update results with computed values
                            for i, idx in enumerate(valid_indices):
                                if valid_mask[i]:
                                    results[idx] = float(correlation_cpu[i])
                else:
                    # Fallback to CPU implementation
                    for i, idx in enumerate(valid_indices):
                        a1 = subregions1[i]
                        a2 = subregions2[i]

                        # Center the data
                        a1 = a1 - np.mean(a1)
                        a2 = a2 - np.mean(a2)

                        # Compute variances
                        var1 = np.sum(a1 * a1)
                        var2 = np.sum(a2 * a2)

                        if var1 > 1e-10 and var2 > 1e-10:
                            # Compute cross correlation
                            numerator = np.sum(a1 * a2)
                            denominator = np.sqrt(var1 * var2)
                            correlation = numerator / denominator

                            if np.isfinite(correlation):
                                results[idx] = float(correlation)

                # Clear memory after each batch
                if batch_size > 1 and (batch_end - batch_start) > batch_size // 2:
                    self._try_clear_memory()

            # Clamp all results to valid range [-1, 1]
            results = [max(-1.0, min(1.0, r)) for r in results]

            elapsed_time = time.time() - start_time
            logging.debug(f"GPU batch_compute_cross_correlation for {len(offsets)} offsets took {elapsed_time:.6f} seconds")

            return results

        except Exception as e:
            logging.warning(f"GPU batch_compute_cross_correlation failed: {str(e)}. Falling back to CPU implementation.")

            # Try to recover by clearing memory
            self._try_clear_memory()

            # Fallback to CPU implementation
            try:
                # Process each offset individually on CPU
                for i, (x, y) in enumerate(offsets):
                    a1 = self.extract_subregion(t1, x, y)
                    a2 = self.extract_subregion(t2, -x, -y)

                    if a1 is not None and a2 is not None and a1.size > 0 and a2.size > 0:
                        a1_flat = a1.ravel().astype(np.float32)
                        a2_flat = a2.ravel().astype(np.float32)

                        # Center the data
                        a1_flat -= np.mean(a1_flat)
                        a2_flat -= np.mean(a2_flat)

                        # Compute variances
                        var1 = np.sum(a1_flat * a1_flat)
                        var2 = np.sum(a2_flat * a2_flat)

                        if var1 > 1e-10 and var2 > 1e-10:
                            # Compute cross correlation
                            numerator = np.sum(a1_flat * a2_flat)
                            denominator = np.sqrt(var1 * var2)
                            correlation = numerator / denominator

                            if np.isfinite(correlation):
                                results[i] = float(correlation)

                elapsed_time = time.time() - start_time
                logging.warning(f"CPU fallback batch_compute_cross_correlation took {elapsed_time:.6f} seconds")

                return results

            except Exception as e2:
                logging.error(f"Both GPU and CPU batch_compute_cross_correlation failed: {str(e2)}")
                return results

    def hill_climb(self, t1: np.ndarray, t2: np.ndarray, initial_x: int, initial_y: int,
                  search_radius: int, cache: np.ndarray = None) -> img_tile.Peak:
        """
        Performs hill climbing to find the peak correlation using GPU acceleration.
        Optimized for both PyTorch and CuPy backends with robust error handling and memory management.

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
        start_time = time.time()

        try:
            # Define the search bounds
            x_min = initial_x - search_radius
            x_max = initial_x + search_radius
            y_min = initial_y - search_radius
            y_max = initial_y + search_radius

            # Initialize the peak
            best_peak = img_tile.Peak(ncc=np.nan, x=initial_x, y=initial_y)

            # If no cache is provided, create one
            if cache is None:
                cache = np.nan * np.ones((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)
            elif cache.shape != ((y_max - y_min + 1), (x_max - x_min + 1)):
                logging.warning(f"Cache shape mismatch in hill_climb: expected {(y_max - y_min + 1, x_max - x_min + 1)}, got {cache.shape}")
                cache = np.nan * np.ones((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

            # Define the directions for hill climbing
            directions = [
                (0, -1),  # North
                (0, 1),   # South
                (1, 0),   # East
                (-1, 0)   # West
            ]

            # Track iterations for logging and safety
            iteration = 0
            max_iterations = 2 * search_radius  # Reasonable upper bound

            # Perform hill climbing
            while iteration < max_iterations:
                iteration += 1
                cur_direction = (0, 0)  # No move

                # Translate current absolute position to 0-based cache indices
                cur_x_idx = best_peak.x - x_min
                cur_y_idx = best_peak.y - y_min

                # Clamp the indices to ensure they fall within cache bounds
                cur_x_idx = max(0, min(cur_x_idx, cache.shape[1] - 1))
                cur_y_idx = max(0, min(cur_y_idx, cache.shape[0] - 1))

                # Get the current NCC value from the cache or compute it
                best_peak.ncc = cache[cur_y_idx, cur_x_idx]
                if np.isnan(best_peak.ncc):
                    best_peak.ncc = self.compute_cross_correlation(t1, t2, best_peak.x, best_peak.y)
                    cache[cur_y_idx, cur_x_idx] = best_peak.ncc

                # Check all directions
                offsets = []
                offset_indices = []

                for dx, dy in directions:
                    new_x = best_peak.x + dx
                    new_y = best_peak.y + dy

                    # Ensure new positions are within search bounds
                    if x_min <= new_x <= x_max and y_min <= new_y <= y_max:
                        new_x_idx = max(0, min(cur_x_idx + dx, cache.shape[1] - 1))
                        new_y_idx = max(0, min(cur_y_idx + dy, cache.shape[0] - 1))

                        # If the NCC value is not in the cache, add this direction to the list
                        if np.isnan(cache[new_y_idx, new_x_idx]):
                            offsets.append((new_x, new_y))
                            offset_indices.append((new_y_idx, new_x_idx))

                # Compute NCC values for all directions in batch if there are any uncached values
                if offsets:
                    try:
                        # Use batch computation for better performance
                        ncc_values = self.batch_compute_cross_correlation(t1, t2, offsets)

                        # Update the cache with the computed values
                        for i, (y_idx, x_idx) in enumerate(offset_indices):
                            if i < len(ncc_values):  # Safety check
                                cache[y_idx, x_idx] = ncc_values[i]
                    except Exception as e:
                        logging.warning(f"Batch computation failed in hill_climb: {str(e)}. Computing individually.")

                        # Fallback to individual computation
                        for i, (new_x, new_y) in enumerate(offsets):
                            if i < len(offset_indices):  # Safety check
                                y_idx, x_idx = offset_indices[i]
                                ncc = self.compute_cross_correlation(t1, t2, new_x, new_y)
                                cache[y_idx, x_idx] = ncc

                # Find the best direction
                best_direction = (0, 0)
                best_ncc = best_peak.ncc

                for dx, dy in directions:
                    new_x = best_peak.x + dx
                    new_y = best_peak.y + dy

                    # Ensure new positions are within search bounds
                    if x_min <= new_x <= x_max and y_min <= new_y <= y_max:
                        new_x_idx = max(0, min(cur_x_idx + dx, cache.shape[1] - 1))
                        new_y_idx = max(0, min(cur_y_idx + dy, cache.shape[0] - 1))

                        ncc = cache[new_y_idx, new_x_idx]
                        if np.isfinite(ncc) and ncc > best_ncc:
                            best_ncc = ncc
                            best_direction = (dx, dy)

                # If no better direction is found, we've reached the peak
                if best_direction == (0, 0):
                    break

                # Update the peak with the best direction
                best_peak.ncc = best_ncc
                best_peak.x += best_direction[0]
                best_peak.y += best_direction[1]

                # Periodically clear GPU memory during long climbs
                if iteration % 10 == 0 and iteration > 0:
                    self._try_clear_memory()

            # If the peak has NaN NCC, set it to a default value
            if np.isnan(best_peak.ncc) or not np.isfinite(best_peak.ncc):
                best_peak.x = int((x_max + x_min) / 2)
                best_peak.y = int((y_max + y_min) / 2)
                best_peak.ncc = -1.0

            # Clamp NCC to valid range [-1, 1]
            best_peak.ncc = max(-1.0, min(1.0, best_peak.ncc))

            elapsed_time = time.time() - start_time
            logging.debug(f"GPU hill_climb took {elapsed_time:.6f} seconds ({iteration} iterations)")

            return best_peak

        except Exception as e:
            logging.warning(f"GPU hill_climb failed: {str(e)}. Falling back to CPU implementation.")

            try:
                # Try to recover by clearing memory
                self._try_clear_memory()

                # Fallback to CPU implementation using the same algorithm
                # Define the search bounds
                x_min = initial_x - search_radius
                x_max = initial_x + search_radius
                y_min = initial_y - search_radius
                y_max = initial_y + search_radius

                # Initialize the peak
                best_peak = img_tile.Peak(ncc=np.nan, x=initial_x, y=initial_y)

                # If no cache is provided, create one
                if cache is None or cache.shape != ((y_max - y_min + 1), (x_max - x_min + 1)):
                    cache = np.nan * np.ones((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

                # Define the directions for hill climbing
                directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W

                # Perform hill climbing
                iteration = 0
                max_iterations = 2 * search_radius

                while iteration < max_iterations:
                    iteration += 1
                    cur_direction = (0, 0)

                    # Get current position in cache
                    cur_x_idx = best_peak.x - x_min
                    cur_y_idx = best_peak.y - y_min
                    cur_x_idx = max(0, min(cur_x_idx, cache.shape[1] - 1))
                    cur_y_idx = max(0, min(cur_y_idx, cache.shape[0] - 1))

                    # Get or compute current NCC
                    best_peak.ncc = cache[cur_y_idx, cur_x_idx]
                    if np.isnan(best_peak.ncc):
                        from MIST.cpu_backend import CPUBackend
                        cpu_backend = CPUBackend()
                        best_peak.ncc = cpu_backend.compute_cross_correlation(t1, t2, best_peak.x, best_peak.y)
                        cache[cur_y_idx, cur_x_idx] = best_peak.ncc

                    # Check all directions
                    best_direction = (0, 0)
                    best_ncc = best_peak.ncc

                    for dx, dy in directions:
                        new_x = best_peak.x + dx
                        new_y = best_peak.y + dy

                        if x_min <= new_x <= x_max and y_min <= new_y <= y_max:
                            new_x_idx = max(0, min(cur_x_idx + dx, cache.shape[1] - 1))
                            new_y_idx = max(0, min(cur_y_idx + dy, cache.shape[0] - 1))

                            ncc = cache[new_y_idx, new_x_idx]
                            if np.isnan(ncc):
                                from MIST.cpu_backend import CPUBackend
                                cpu_backend = CPUBackend()
                                ncc = cpu_backend.compute_cross_correlation(t1, t2, new_x, new_y)
                                cache[new_y_idx, new_x_idx] = ncc

                            if np.isfinite(ncc) and ncc > best_ncc:
                                best_ncc = ncc
                                best_direction = (dx, dy)

                    # If no better direction, we're done
                    if best_direction == (0, 0):
                        break

                    # Update peak
                    best_peak.ncc = best_ncc
                    best_peak.x += best_direction[0]
                    best_peak.y += best_direction[1]

                # Handle invalid peak
                if np.isnan(best_peak.ncc) or not np.isfinite(best_peak.ncc):
                    best_peak.x = int((x_max + x_min) / 2)
                    best_peak.y = int((y_max + y_min) / 2)
                    best_peak.ncc = -1.0

                elapsed_time = time.time() - start_time
                logging.warning(f"CPU fallback hill_climb took {elapsed_time:.6f} seconds ({iteration} iterations)")

                return best_peak

            except Exception as e2:
                logging.error(f"Both GPU and CPU hill_climb failed: {str(e2)}")

                # Return a default peak as last resort
                default_peak = img_tile.Peak(ncc=-1.0, x=initial_x, y=initial_y)
                return default_peak
