# plan_01e_gpu_backend_part1.md
## Component: GPU Backend Implementation - Core Structure

### Objective
Create a robust, high-performance GPU-accelerated backend implementation of the ComputeBackend interface that leverages CUDA for faster image correlation and refinement operations. This backend should be fully interchangeable with the CPU backend while providing significant performance improvements for compute-intensive operations, with comprehensive error handling and memory management.

### Plan
1. Implement a GPUBackend class that conforms to the ComputeBackend interface
   - Create a new file `gpu_backend.py` that implements all required methods
   - Support both PyTorch and CuPy as GPU acceleration libraries with dynamic selection
   - Implement comprehensive device management and memory handling
   - Add detailed initialization checks and error reporting

2. Implement core utility methods for GPU operations
   - Create methods for transferring data between CPU and GPU
   - Implement memory management and tracking
   - Add error handling and recovery mechanisms
   - Support both PyTorch and CuPy backends

3. Implement basic image processing operations
   - Create GPU-accelerated versions of extract_subregion
   - Ensure proper error handling and fallback mechanisms
   - Optimize for performance while maintaining accuracy

### Findings
The MIST codebase uses normalized cross-correlation (NCC) extensively for image alignment, with the primary computational hotspots being:

1. **compute_cross_correlation**: This method computes the NCC between two image tiles at a given offset. It's called repeatedly during hill climbing and is a prime candidate for GPU acceleration.

2. **cross_correlation**: This method computes the NCC between two arrays and is called by compute_cross_correlation. It involves matrix operations that can be efficiently parallelized on a GPU.

3. **extract_subregion**: This method extracts a subregion from an image based on a translation. It's a simple array slicing operation but is called frequently.

The GPU backend needs to be designed with robust error handling and memory management to handle large images and gracefully fall back to CPU when needed.

### Implementation Draft

```python
import numpy as np
import logging
import time
import MIST.img_tile as img_tile
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
        
        Args:
            t1: The image tile a sub-region is being extracted from
            x: The x component of the translation
            y: The y component of the translation
            
        Returns:
            The extracted sub-region, or None if there's no overlap
        """
        # This operation is simple enough that it's more efficient to do it on the CPU
        # to avoid the overhead of transferring data to and from the GPU
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
```
