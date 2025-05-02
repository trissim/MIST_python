# plan_01e_gpu_backend_part2.md
## Component: GPU Backend Implementation - Cross Correlation

### Objective
Implement GPU-accelerated cross correlation operations for the MIST image stitching pipeline. These operations are the most computationally intensive part of the pipeline and will benefit significantly from GPU acceleration.

### Plan
1. Implement GPU-accelerated cross correlation
   - Create optimized versions for both PyTorch and CuPy
   - Add robust error handling and fallback mechanisms
   - Optimize for performance with large images
   - Ensure numerical stability and accuracy

2. Implement batch processing for cross correlation
   - Create methods for computing multiple correlations in parallel
   - Optimize memory usage for batch operations
   - Add adaptive batch sizing based on available GPU memory
   - Ensure proper error handling and recovery

3. Implement compute_cross_correlation and related methods
   - Create GPU-accelerated versions of all correlation methods
   - Ensure compatibility with the existing API
   - Add performance monitoring and logging
   - Optimize for different image sizes and shapes

### Findings
The cross correlation operations in MIST are the most computationally intensive part of the pipeline. They involve computing the normalized cross correlation (NCC) between two image regions, which requires multiple matrix operations that can be efficiently parallelized on a GPU.

The current implementation in PCIAM.cross_correlation and PCIAM.compute_cross_correlation uses NumPy operations that run on the CPU. By moving these operations to the GPU, we can achieve significant speedups, especially for large images and batch operations.

### Implementation Draft

```python
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
```
