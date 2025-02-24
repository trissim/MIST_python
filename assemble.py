import os
import numpy as np
import skimage.io

def create_linear_weight_mask(H, W, margin_ratio=0.05):
    """
    Create a 2D weight mask of shape (H, W) that linearly ramps from 0 at the edges
    to 1 in the center. The margin_ratio specifies the fraction of the tile dimension
    over which to ramp.
    """
    # Determine margins in each dimension
    margin_y = int(np.floor(H * margin_ratio))
    margin_x = int(np.floor(W * margin_ratio))

    # Create a 1D weight for Y direction
    weight_y = np.ones(H, dtype=np.float32)
    if margin_y > 0:
        # Ramp up from 0 to 1 at top
        ramp_y_top = np.linspace(0, 1, margin_y, endpoint=False)
        # Ramp down from 1 to 0 at bottom
        ramp_y_bottom = np.linspace(1, 0, margin_y, endpoint=False)
        weight_y[:margin_y] = ramp_y_top
        weight_y[-margin_y:] = ramp_y_bottom

    # Create a 1D weight for X direction
    weight_x = np.ones(W, dtype=np.float32)
    if margin_x > 0:
        ramp_x_left = np.linspace(0, 1, margin_x, endpoint=False)
        ramp_x_right = np.linspace(1, 0, margin_x, endpoint=False)
        weight_x[:margin_x] = ramp_x_left
        weight_x[-margin_x:] = ramp_x_right

    # The overall 2D mask is the outer product of the 1D weights
    mask = np.outer(weight_y, weight_x)
    return mask

def assemble_image(global_positions_filepath, images_dirpath, output_filepath, img_names=None, binning=1, margin_ratio=0.1):
    """
    Assemble a stitched image using global positions read from a text file and
    blend overlapping regions with a linear weight mask.
    
    The global positions file is expected to have lines in the format:
    
        "File: filename; ...; Pos: (x, y); ..."
    
    The positions are multiplied by `binning` to scale.
    
    Parameters:
      - global_positions_filepath: path to the text file with global positions.
      - images_dirpath: directory containing the image tiles.
      - output_filepath: path where the stitched image will be saved.
      - img_names: optional list of image file names (if None, they are read from the file).
      - binning: integer binning factor to scale positions.
      - margin_ratio: fraction of tile dimension over which to linearly feather the tile edges.
    
    Returns:
      None. The stitched image is saved to output_filepath.
    """
    # Ensure output directory exists
    parent, _ = os.path.split(output_filepath)
    if not os.path.exists(parent):
        os.makedirs(parent)
    
    if not os.path.exists(global_positions_filepath):
        raise RuntimeError(f'Missing global positions file: {global_positions_filepath}')
    
    # Parse global positions
    add_names = False
    if img_names is None:
        add_names = True
        img_names = []
    pixel_x_position = []
    pixel_y_position = []
    
    with open(global_positions_filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            toks = line.split(';')
            # Expect first token: "File: filename"
            fn_tok = toks[0]
            fn = fn_tok.split(':')[1].strip()
            if add_names:
                img_names.append(fn)
            # Expect third token: "Pos: (x, y)"
            pos_tok = toks[2]
            pos_pair = pos_tok.split(':')[1].strip()
            # Remove parentheses
            pos_pair = pos_pair.replace('(', '').replace(')', '')
            pos_parts = pos_pair.split(',')
            x = int(pos_parts[0].strip()) * binning
            y = int(pos_parts[1].strip()) * binning
            pixel_x_position.append(x)
            pixel_y_position.append(y)
    
    # Verify images exist
    if not os.path.exists(images_dirpath):
        raise RuntimeError(f'Images directory does not exist: {images_dirpath}')
    for fn in img_names:
        if not os.path.exists(os.path.join(images_dirpath, fn)):
            raise RuntimeError(f'Image {fn} expected based on global positions file is missing from the directory.')
    
    # Read the first tile to determine tile shape and dtype
    first_tile = skimage.io.imread(os.path.join(images_dirpath, img_names[0]))
    tile_shape = first_tile.shape
    tile_h, tile_w = tile_shape[:2]
    dtype = first_tile.dtype
    
    # Compute final stitched image dimensions
    stitched_img_h = tile_h + np.max(pixel_y_position)
    stitched_img_w = tile_w + np.max(pixel_x_position)
    
    # Create float32 accumulators for pixel values and weights
    if len(tile_shape) == 2:
        acc = np.zeros((stitched_img_h, stitched_img_w), dtype=np.float32)
        weight_acc = np.zeros((stitched_img_h, stitched_img_w), dtype=np.float32)
    else:
        n_channels = tile_shape[2]
        acc = np.zeros((stitched_img_h, stitched_img_w, n_channels), dtype=np.float32)
        weight_acc = np.zeros((stitched_img_h, stitched_img_w, n_channels), dtype=np.float32)
    
    print(f'Creating blank stitched image of size: ({stitched_img_h}, {stitched_img_w}, {tile_shape[2] if len(tile_shape)==3 else 1})')
    
    # Process each tile
    for i, fn in enumerate(img_names):
        x = pixel_x_position[i]
        y = pixel_y_position[i]
        print(f'Placing image {i+1}/{len(img_names)}: {fn} at ({x}, {y})')
        tile = skimage.io.imread(os.path.join(images_dirpath, fn))
        if tile.shape != tile_shape:
            raise RuntimeError(f'All images must have the same shape. Image {fn} is {tile.shape}, expected {tile_shape}.')
        if tile.dtype != dtype:
            raise RuntimeError(f'Image {fn} has type {tile.dtype}, expected {dtype}.')
        
        # Create a linear weight mask for this tile
        mask = create_linear_weight_mask(tile_h, tile_w, margin_ratio=margin_ratio)
        # If multi-channel, expand mask along the channel dimension
        if tile.ndim == 3:
            mask = mask[..., np.newaxis]
        
        # Convert tile to float32 for blending
        tile_float = tile.astype(np.float32)
        weighted_tile = tile_float * mask
        
        # Determine region on the canvas (using integer indices)
        y_start = int(y)
        x_start = int(x)
        y_end = y_start + tile_h
        x_end = x_start + tile_w
        
        # Accumulate weighted pixel values and mask weights
        acc[y_start:y_end, x_start:x_end] += weighted_tile
        weight_acc[y_start:y_end, x_start:x_end] += mask
    
    # Compute the final blended image (avoid division by zero)
    safe_weight = np.where(weight_acc == 0, 1, weight_acc)
    blended = acc / safe_weight
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
    else:
        max_val = np.finfo(dtype).max
    blended = np.clip(blended, 0, max_val).astype(dtype)   # Convert to original dtype (clamp as needed)

    print('Saving stitched image to disk')
    skimage.io.imsave(output_filepath, blended, plugin=None, tile=(1024, 1024), check_contrast=False)
    print(f'Stitched image saved to {output_filepath}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to assemble MIST stitched image')
    parser.add_argument('--global-positions-filepath', type=str, required=True, help='Filepath to the global positions file generated by MIST.')
    parser.add_argument('--images-dirpath', type=str, required=True, help='Dirpath (directory) where the source images exists.')
    parser.add_argument('--output-filepath', type=str, required=True, help='Filepath where to save the resulting stitched image.')

    args = parser.parse_args()
    global_positions_filepath = args.global_positions_filepath
    images_dirpath = args.images_dirpath
    output_filepath = args.output_filepath

    assemble_image(global_positions_filepath, images_dirpath, output_filepath)
