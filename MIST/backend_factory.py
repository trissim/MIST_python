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
