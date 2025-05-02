# plan_01b_gpu_hooks_utils.md
## Component: GPU Detection Utilities and Backend Factory

### Objective
Extend utils.py with GPU detection capabilities and create a factory function for backend instantiation.

### Implementation Draft

#### 1. Extend utils.py with GPU Detection Capabilities

We'll add the following functions to utils.py to detect and manage GPU resources:

```python
# GPU detection and management functions
def is_cuda_available():
    """
    Check if CUDA is available on the system.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import cupy
            return True
        except ImportError:
            return False

def get_gpu_count():
    """
    Get the number of available GPU devices.
    
    Returns:
        int: Number of available GPU devices, or 0 if no GPUs are available
    """
    if not is_cuda_available():
        return 0
    
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount()
        except:
            return 0

def get_gpu_info(device_id=0):
    """
    Get information about a specific GPU device.
    
    Args:
        device_id (int): The ID of the GPU device to query
        
    Returns:
        dict: Dictionary containing GPU information, or None if the device is not available
    """
    if not is_cuda_available() or device_id >= get_gpu_count():
        return None
    
    try:
        import torch
        if device_id < torch.cuda.device_count():
            return {
                'name': torch.cuda.get_device_name(device_id),
                'memory_total': torch.cuda.get_device_properties(device_id).total_memory,
                'memory_available': torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
            }
    except ImportError:
        try:
            import cupy
            device = cupy.cuda.Device(device_id)
            return {
                'name': device.name,
                'memory_total': device.mem_info[0],
                'memory_available': device.mem_info[1]
            }
        except:
            pass
    
    return None

def select_gpu_device(args):
    """
    Select the appropriate GPU device based on command-line arguments.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        int: The selected GPU device ID, or -1 if no GPU should be used
    """
    # If GPU usage is not requested, return -1
    if not hasattr(args, 'use_gpu') or not args.use_gpu:
        return -1
    
    # If CUDA is not available, log a warning and return -1
    if not is_cuda_available():
        logging.warning("GPU usage requested but CUDA is not available. Falling back to CPU.")
        return -1
    
    # Get the number of available GPUs
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        logging.warning("GPU usage requested but no GPUs are available. Falling back to CPU.")
        return -1
    
    # If a specific GPU device is requested, check if it's valid
    if hasattr(args, 'gpu_device') and args.gpu_device is not None:
        if args.gpu_device >= gpu_count:
            logging.warning(f"Requested GPU device {args.gpu_device} is not available. "
                           f"Only {gpu_count} devices are available. Falling back to device 0.")
            return 0
        return args.gpu_device
    
    # If no specific device is requested, use device 0
    return 0
```

#### 2. Create the Backend Factory

Now, let's create a factory function to instantiate the appropriate backend based on the command-line arguments:

```python
import logging
import argparse
from MIST.compute_backend import ComputeBackend
from MIST.cpu_backend import CPUBackend
import MIST.utils as utils

def create_compute_backend(args: argparse.Namespace) -> ComputeBackend:
    """
    Factory function to create the appropriate compute backend based on command-line arguments.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        ComputeBackend: An instance of a ComputeBackend implementation
    """
    # Check if GPU usage is requested
    if hasattr(args, 'use_gpu') and args.use_gpu:
        # Select the appropriate GPU device
        device_id = utils.select_gpu_device(args)
        
        # If a valid GPU device is selected, try to create a GPU backend
        if device_id >= 0:
            try:
                # Import the GPU backend only if needed
                from MIST.gpu_backend import GPUBackend
                
                # Get information about the selected GPU
                gpu_info = utils.get_gpu_info(device_id)
                if gpu_info:
                    logging.info(f"Using GPU backend with device {device_id}: {gpu_info['name']}")
                    return GPUBackend(device_id)
                else:
                    logging.warning(f"Could not get information about GPU device {device_id}. Falling back to CPU.")
            except ImportError:
                logging.warning("GPU backend requested but GPU backend module is not available. Falling back to CPU.")
            except Exception as e:
                logging.warning(f"Error initializing GPU backend: {str(e)}. Falling back to CPU.")
    
    # If we reach here, use the CPU backend
    logging.info("Using CPU backend")
    return CPUBackend()
```
