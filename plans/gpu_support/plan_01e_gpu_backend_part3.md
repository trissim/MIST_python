# plan_01e_gpu_backend_part3.md
## Component: GPU Backend Implementation - Peak Finding and Hill Climbing

### Objective
Implement GPU-accelerated peak finding and hill climbing algorithms for the MIST image stitching pipeline. These operations are used to refine the alignment between image tiles and can benefit significantly from GPU acceleration.

### Plan
1. Implement GPU-accelerated peak cross correlation methods
   - Create optimized versions of peak_cross_correlation_worker
   - Implement GPU-accelerated versions of peak_cross_correlation_lr and peak_cross_correlation_ud
   - Ensure compatibility with the existing API
   - Add performance monitoring and logging

2. Implement GPU-accelerated hill climbing
   - Create an optimized version of the hill climbing algorithm that leverages GPU parallelism
   - Use batch processing to evaluate multiple directions simultaneously
   - Implement memory management for large images
   - Add robust error handling and fallback mechanisms

3. Implement compute_pciam method
   - Create a GPU-accelerated version of the phase correlation method
   - Optimize FFT operations using GPU libraries
   - Ensure proper error handling and recovery
   - Add performance monitoring and logging

### Findings
The peak finding and hill climbing algorithms in MIST are used to refine the alignment between image tiles. They involve computing the cross correlation between two image regions at multiple offsets and finding the offset with the highest correlation. These operations can be significantly accelerated on a GPU, especially when using batch processing to evaluate multiple offsets simultaneously.

The current implementation in PCIAM and translation_refinement.py uses CPU-based operations that can be slow for large images. By moving these operations to the GPU and using batch processing, we can achieve significant speedups.

### Implementation Draft

```python
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
```
