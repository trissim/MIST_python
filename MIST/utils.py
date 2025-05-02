import logging
import argparse
import os
import multiprocessing

def get_num_workers() -> int:
    """
    Get the number of available CPU cores for parallel processing.

    Returns:
        int: Number of available CPU cores
    """
    return max(1, multiprocessing.cpu_count() - 1)

def is_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        # Try PyTorch first
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            # Try CuPy next
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount() > 0
        except ImportError:
            return False
        except Exception:
            return False

def get_gpu_count() -> int:
    """
    Get the number of available GPU devices.

    Returns:
        int: Number of available GPU devices, or 0 if no GPUs are available
    """
    try:
        # Try PyTorch first
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    try:
        # Try CuPy next
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except ImportError:
        return 0
    except Exception:
        return 0

    return 0

def get_gpu_info(device_id: int) -> dict:
    """
    Get information about a specific GPU device.

    Args:
        device_id (int): The ID of the GPU device

    Returns:
        dict: Dictionary containing GPU information, or None if the device is not available
    """
    try:
        # Try PyTorch first
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            return {
                'name': props.name,
                'memory_total': props.total_memory,
                'memory_available': props.total_memory,  # Approximation
                'compute_capability': f"{props.major}.{props.minor}"
            }
    except ImportError:
        pass

    try:
        # Try CuPy next
        import cupy as cp
        if device_id < cp.cuda.runtime.getDeviceCount():
            with cp.cuda.Device(device_id):
                attrs = cp.cuda.runtime.deviceGetAttributes(device_id)
                free, total = cp.cuda.runtime.memGetInfo()
                return {
                    'name': cp.cuda.runtime.deviceGetName(device_id),
                    'memory_total': total,
                    'memory_available': free,
                    'compute_capability': f"{attrs['computeCapabilityMajor']}.{attrs['computeCapabilityMinor']}"
                }
    except ImportError:
        return None
    except Exception:
        return None

    return None

def select_gpu_device(args: argparse.Namespace) -> int:
    """
    Select the appropriate GPU device based on command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        int: The ID of the selected GPU device, or -1 if no suitable device is found
    """
    # Check if CUDA is available
    if not is_cuda_available():
        logging.warning("CUDA is not available on this system. Cannot use GPU.")
        return -1

    # Get the number of available GPU devices
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        logging.warning("No GPU devices found. Cannot use GPU.")
        return -1

    # Check if a specific GPU device is requested
    if hasattr(args, 'gpu_device') and args.gpu_device is not None:
        device_id = args.gpu_device
        if device_id >= gpu_count:
            logging.warning(f"Requested GPU device {device_id} is not available. Only {gpu_count} devices found.")
            return -1
        return device_id

    # Check if a specific GPU device is specified in the environment
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        try:
            device_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
            if device_id >= gpu_count:
                logging.warning(f"GPU device {device_id} specified in CUDA_VISIBLE_DEVICES is not available. Only {gpu_count} devices found.")
                return -1
            return device_id
        except ValueError:
            # CUDA_VISIBLE_DEVICES might contain a comma-separated list or other format
            # In this case, just use the first available device
            pass

    # If no specific device is requested, use the first available device
    return 0
