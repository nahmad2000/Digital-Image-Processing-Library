"""
Core image processing functionality for DIP-Lib.
Contains implementations of image processing modules.
Applying feedback from Gemini, ChatGPT, GROK, and User.
Correcting remove_noise signature mismatch.
"""

import cv2
import numpy as np
# Note: skimage.metrics were used previously but calculate_metrics is now in utils.py
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
import pandas as pd
# from utils import calculate_metrics # calculate_metrics is called in main.py, not needed here directly

# --- Image Sharpening ---
def sharpen_image(image, kernel_size=5, weight=1.5):
    """Sharpen an image using Unsharp Masking."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")
    if image.ndim not in [2, 3]:
        raise ValueError(f"Invalid image dimensions: {image.ndim}. Must be 2D (grayscale) or 3D (color).")

    # Ensure kernel size is odd
    if not isinstance(kernel_size, int) or kernel_size <= 0:
         raise ValueError("kernel_size must be a positive integer.")
    if kernel_size % 2 == 0:
        kernel_size += 1

    if not isinstance(weight, (float, int)) or weight <= 0:
         raise ValueError("weight must be a positive number.")

    try:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Calculate the sharpened image using weighted addition
        sharpened = cv2.addWeighted(image, 1.0 + weight, blurred, -weight, 0)

        # Clip values to valid range [0, 255] and ensure correct dtype
        sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)

        return sharpened
    except cv2.error as e:
        raise ValueError(f"OpenCV error during sharpening: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during sharpening: {e}")


# --- Thresholding ---
def apply_threshold(image, threshold_value=127, threshold_type='binary',
                    use_adaptive=False, adaptive_method='mean', block_size=11, C=2):
    """Apply global, adaptive, Otsu, or Triangle thresholding to an image."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")

    # Convert to grayscale if color image
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
         raise ValueError(f"Unsupported image shape for thresholding: {image.shape}")

    # Map threshold type strings to OpenCV constants
    thresh_types = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV,
        'trunc': cv2.THRESH_TRUNC,
        'tozero': cv2.THRESH_TOZERO,
        'tozero_inv': cv2.THRESH_TOZERO_INV,
        'otsu': cv2.THRESH_OTSU, # Added Otsu
        'triangle': cv2.THRESH_TRIANGLE # Added Triangle
    }

    # Map adaptive method strings to OpenCV constants
    adaptive_methods = {
        'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
        'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }

    try:
        if use_adaptive:
            if not isinstance(block_size, int) or block_size <= 1:
                 raise ValueError("block_size for adaptive thresholding must be an integer > 1.")
            if block_size % 2 == 0:
                 block_size += 1 # Ensure block size is odd
            if not isinstance(C, (int, float)):
                 raise ValueError("Constant C for adaptive thresholding must be a number.")

            adaptive_thresh_type_str = threshold_type if threshold_type in ['binary', 'binary_inv'] else 'binary'
            adaptive_thresh_type = thresh_types[adaptive_thresh_type_str] # Adaptive usually uses binary or inv
            adapt_method = adaptive_methods.get(adaptive_method, cv2.ADAPTIVE_THRESH_MEAN_C)

            thresholded = cv2.adaptiveThreshold(gray, 255, adapt_method,
                                                adaptive_thresh_type, block_size, C)
        else:
            # Global thresholding (including Otsu and Triangle)
            thresh_type_flag = thresh_types.get(threshold_type.lower()) # Use lower() for safety
            if thresh_type_flag is None:
                raise ValueError(f"Invalid threshold type: {threshold_type}")

            # For Otsu and Triangle, the input threshold value is ignored (use 0)
            # Combine with binary or binary_inv
            base_thresh_type = cv2.THRESH_BINARY # Default base type
            if 'inv' in threshold_type:
                 base_thresh_type = cv2.THRESH_BINARY_INV

            if threshold_type.lower() in ['otsu', 'triangle']:
                 thresh_type_flag |= base_thresh_type # Combine e.g. THRESH_OTSU | THRESH_BINARY
                 threshold_value_to_use = 0 # Ignored by Otsu/Triangle
            else:
                 threshold_value_to_use = threshold_value

            if not isinstance(threshold_value_to_use, int) or not (0 <= threshold_value_to_use <= 255):
                 # Only validate if not Otsu/Triangle
                 if threshold_type.lower() not in ['otsu', 'triangle']:
                      raise ValueError("threshold_value must be an integer between 0 and 255.")

            _, thresholded = cv2.threshold(gray, threshold_value_to_use, 255, thresh_type_flag)

        return thresholded

    except cv2.error as e:
        raise ValueError(f"OpenCV error during thresholding: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during thresholding: {e}")


# --- Downsampling & Interpolation ---
def downsample_interpolate(image, downsample_method='area', scale_factor=0.5,
                           interpolation_method='bicubic'):
    """Downsample an image and then upsample it using specified methods."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")
    if not (0 < scale_factor < 1):
         raise ValueError("scale_factor must be between 0 and 1 (exclusive).")

    h, w = image.shape[:2]

    downsample_methods = {'simple': cv2.INTER_NEAREST, 'antialias': cv2.INTER_AREA, 'area': cv2.INTER_AREA}
    interpolation_methods = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}

    if downsample_method not in downsample_methods: raise ValueError(f"Invalid downsample method: {downsample_method}")
    if interpolation_method not in interpolation_methods: raise ValueError(f"Invalid interpolation method: {interpolation_method}")

    try:
        new_w = max(1, int(w * scale_factor)) # Ensure at least 1 pixel
        new_h = max(1, int(h * scale_factor))

        downsampled = cv2.resize(image, (new_w, new_h), interpolation=downsample_methods[downsample_method])
        upsampled = cv2.resize(downsampled, (w, h), interpolation=interpolation_methods[interpolation_method])

        # Calculate metrics in main.py now, not here
        # metrics = calculate_metrics(image, upsampled, metrics=['psnr', 'ssim']) # Moved

        return {
            'image': upsampled,
            'downsampled': downsampled,
            # 'metrics': metrics, # Moved to main.py
            'params': {'downsample_method': downsample_method, 'scale_factor': scale_factor, 'interpolation_method': interpolation_method}
        }
    except cv2.error as e:
        raise ValueError(f"OpenCV error during resizing: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during resizing: {e}")


# --- Geometric Transformations ---
def geometric_transform(image, rotation=0, scale_x=1.0, scale_y=1.0,
                        translate_x=0, translate_y=0, shear_x=0, shear_y=0):
    """Apply geometric transformations to an image."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    try:
        # Rotation and Scaling Matrix
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        M[0, 0] *= scale_x
        M[1, 1] *= scale_y

        # Apply Translation
        M[0, 2] += translate_x
        M[1, 2] += translate_y

        # Combine with Shear if necessary
        if shear_x != 0 or shear_y != 0:
            shear_matrix = np.array([ [1, shear_x, 0], [shear_y, 1, 0] ], dtype=np.float32)
            # Need to convert M to 3x3, multiply, then convert back to 2x3
            M_3x3 = np.vstack([M, [0, 0, 1]])
            shear_M_3x3 = np.array([ [1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float32)

            # Combine: Apply shear first, then rotation/scale/translate
            # Or vice-versa depending on desired effect. This applies shear *before* others.
            # combined_3x3 = M_3x3 @ shear_M_3x3 # Shear -> Others
            # Let's apply shear *after* rotation/scale/translation for consistency with some tools
            combined_3x3 = shear_M_3x3 @ M_3x3 # Others -> Shear

            transform_matrix = combined_3x3[:2, :]
        else:
            transform_matrix = M

        # Apply the transformation
        # Using BORDER_REFLECT as a reasonable default for filling empty areas
        result = cv2.warpAffine(image, transform_matrix, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return result
    except cv2.error as e:
        raise ValueError(f"OpenCV error during geometric transformation: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during geometric transformation: {e}")


# --- Noise Addition ---
def add_noise(image, noise_type='gaussian', amount=0.05):
    """Add synthetic noise to an image."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")
    if noise_type not in ['gaussian', 'salt_pepper']:
         raise ValueError(f"Unknown noise type: {noise_type}")
    if not isinstance(amount, float) or not (0 < amount <= 1.0): # amount range might differ based on noise type interpretation
         # For salt & pepper, amount is density. For gaussian, it's variance.
         # Let's allow up to 1.0 for variance, though 0.2 is high.
         raise ValueError("Noise amount must be a float > 0 and <= 1.0.")

    try:
        # random_noise works with float images in range [0, 1]
        if image.dtype != np.uint8:
             # If already float, assume it's [0, 1] or [0, 255]. Normalize if needed.
             if image.max() > 1.0:
                  image_float = image.astype(np.float32) / 255.0
             else:
                  image_float = image.astype(np.float32)
        else:
             image_float = image.astype(np.float32) / 255.0

        if noise_type == 'gaussian':
            # 'amount' is interpreted as variance
            noisy = random_noise(image_float, mode='gaussian', var=amount, clip=True) # clip=True ensures output is [0,1]
        elif noise_type == 'salt_pepper':
            # 'amount' is interpreted as density
            noisy = random_noise(image_float, mode='s&p', amount=amount, clip=True) # clip is default here

        # Convert back to original dtype (usually uint8)
        noisy_image = (noisy * 255).astype(image.dtype)

        return noisy_image
    except Exception as e:
        # Catch potential errors from skimage
        raise RuntimeError(f"Error adding {noise_type} noise: {e}")


# --- Noise Removal ---
# **** CORRECTED FUNCTION SIGNATURE AND BODY ****
def remove_noise(image, filter_type='gaussian', kernel_size=5, sigma=1.5, h=10):
     """
     Apply noise removal filter to an image.
     Accepts parameters directly as keyword arguments.
     """
     if not isinstance(image, np.ndarray):
         raise ValueError("Invalid input: image must be a numpy array.")
     if filter_type not in ['gaussian', 'median', 'nlm']:
          raise ValueError(f"Unknown filter type: {filter_type}")

     try:
         filtered = image.copy() # Start with a copy

         if filter_type == 'gaussian':
             # Use kernel_size and sigma directly from arguments
             if not isinstance(kernel_size, int) or kernel_size <= 0: raise ValueError("kernel_size must be a positive integer.")
             if kernel_size % 2 == 0: kernel_size += 1
             if not isinstance(sigma, (float, int)) or sigma <= 0: raise ValueError("sigma must be a positive number.")
             filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

         elif filter_type == 'median':
             # Use kernel_size directly from arguments
             if not isinstance(kernel_size, int) or kernel_size <= 0: raise ValueError("kernel_size must be a positive integer.")
             if kernel_size % 2 == 0: kernel_size += 1
             filtered = cv2.medianBlur(image, kernel_size)

         elif filter_type == 'nlm':
             # Use h directly from arguments
             if not isinstance(h, (int, float)) or h <= 0: raise ValueError("Filter strength 'h' for NLM must be a positive number.")

             # NLM implementation from skimage handles color/grayscale detection
             # It expects float images in [0, 1] range typically
             if image.dtype == np.uint8:
                 image_float = image.astype(np.float32) / 255.0
                 was_uint8 = True
             elif image.max() > 1.0: # Assume float is [0, 255]
                  image_float = image.astype(np.float32) / 255.0
                  was_uint8 = False # Or original float type
             else: # Assume float is already [0, 1]
                  image_float = image.astype(np.float32)
                  was_uint8 = False

             is_color = len(image_float.shape) == 3

             # Estimate noise standard deviation
             sigma_est = estimate_sigma(image_float, channel_axis=-1 if is_color else None)
             if is_color and isinstance(sigma_est, (tuple, list, np.ndarray)): # Check if multiple values returned for color
                  sigma_est = np.mean(sigma_est) # Use average sigma for color

             # Adjust h based on estimated sigma. h acts as a multiplier.
             h_adjusted = h * sigma_est

             # Parameters for denoise_nl_means
             patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=-1 if is_color else None) # Example params

             # Handle multichannel kwarg based on skimage version (try/except or version check)
             try:
                 # Newer skimage expects channel_axis
                 filtered_float = denoise_nl_means(image_float, h=h_adjusted, sigma=sigma_est,
                                                  fast_mode=True, **patch_kw)
             except TypeError:
                 # Older skimage might expect multichannel boolean
                 patch_kw.pop('channel_axis', None) # Remove if not supported
                 filtered_float = denoise_nl_means(image_float, h=h_adjusted, sigma=sigma_est,
                                                  fast_mode=True, multichannel=is_color, # Use multichannel instead
                                                  patch_size=5, patch_distance=6) # Re-add patch params


             # Convert back to original range and type
             if was_uint8:
                 filtered = np.clip(filtered_float * 255, 0, 255).astype(np.uint8)
             else:
                 # Handle potential scaling if original was float [0, 255]
                 if image.max() > 1.0 and image.dtype != np.uint8:
                      filtered = np.clip(filtered_float * 255, 0, 255).astype(image.dtype)
                 else: # Assume original was float [0, 1]
                      filtered = np.clip(filtered_float, 0, 1).astype(image.dtype)

         return filtered

     except cv2.error as e:
         raise ValueError(f"OpenCV error during noise removal ({filter_type}): {e}")
     except Exception as e:
         raise RuntimeError(f"Unexpected error during noise removal ({filter_type}): {e}")


# --- Image Enhancement ---
def enhance_image(image, gamma=1.0, use_clahe=False, clip_limit=2.0,
                 tile_grid_size=(8, 8), use_equalization=False):
    """Enhance an image using gamma correction, histogram equalization, and/or CLAHE."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")

    enhanced = image.copy()

    try:
        # 1. Apply gamma correction
        if not isinstance(gamma, (float, int)) or gamma <= 0: raise ValueError("Gamma must be a positive number.")
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            # Handle potential division by zero if input is 0
            table = np.array([((i / 255.0) ** inv_gamma) * 255 if i > 0 else 0 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)

        # 2. Apply Histogram Equalization or CLAHE
        is_color = len(enhanced.shape) == 3 and enhanced.shape[2] == 3

        if use_equalization and use_clahe:
             # Typically, you'd use one or the other for contrast enhancement. Warn user?
             # Let's prioritize CLAHE if both are selected, as it's often better.
             use_equalization = False
             # print("Warning: Both Equalization and CLAHE selected. Applying only CLAHE.")

        if use_clahe:
            if not isinstance(clip_limit, (float, int)) or clip_limit < 0: raise ValueError("CLAHE clip_limit must be non-negative.")
            if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2 or not all(isinstance(x, int) and x > 0 for x in tile_grid_size):
                raise ValueError("CLAHE tile_grid_size must be a tuple of two positive integers.")

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

            if is_color:
                # Apply CLAHE to L channel of LAB space
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_clahe = clahe.apply(l)
                lab = cv2.merge((l_clahe, a, b))
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else: # Grayscale
                enhanced = clahe.apply(enhanced)

        elif use_equalization:
             if is_color:
                 # Apply Histogram Equalization to V channel of HSV space (often better than per-channel BGR)
                 hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
                 h, s, v = cv2.split(hsv)
                 v_equalized = cv2.equalizeHist(v)
                 hsv = cv2.merge((h, s, v_equalized))
                 enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                 # Alternative: Equalize each BGR channel (can cause color shifts)
                 # b, g, r = cv2.split(enhanced)
                 # b = cv2.equalizeHist(b)
                 # g = cv2.equalizeHist(g)
                 # r = cv2.equalizeHist(r)
                 # enhanced = cv2.merge((b, g, r))
             else: # Grayscale
                 enhanced = cv2.equalizeHist(enhanced)

        return enhanced
    except cv2.error as e:
        raise ValueError(f"OpenCV error during enhancement: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during enhancement: {e}")


# --- Lighting Correction ---
def correct_lighting(image, method='spatial', kernel_size=51, gamma=1.5,
                    cutoff_low=0.5, cutoff_high=2.0):
    """Correct uneven lighting using spatial (blur-based) or frequency (homomorphic) methods."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")
    if method not in ['spatial', 'frequency']:
         raise ValueError(f"Unknown lighting correction method: {method}")

    is_color = len(image.shape) == 3 and image.shape[2] == 3

    try:
        if method == 'spatial':
            if not isinstance(kernel_size, int) or kernel_size <= 1: raise ValueError("Spatial kernel_size must be an integer > 1.")
            if kernel_size % 2 == 0: kernel_size += 1

            if is_color:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_float = l.astype(np.float32) / 255.0
                background = cv2.GaussianBlur(l_float, (kernel_size, kernel_size), 0)
                # Avoid division by zero or near-zero
                corrected_l_float = l_float / (background + 1e-5)
                # Normalize to prevent blowup and map back to [0, 1]
                cv2.normalize(corrected_l_float, corrected_l_float, 0, 1, cv2.NORM_MINMAX)
                corrected_l = (corrected_l_float * 255).astype(np.uint8)
                lab = cv2.merge((corrected_l, a, b))
                corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else: # Grayscale
                image_float = image.astype(np.float32) / 255.0
                background = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 0)
                corrected_float = image_float / (background + 1e-5)
                cv2.normalize(corrected_float, corrected_float, 0, 1, cv2.NORM_MINMAX)
                corrected = (corrected_float * 255).astype(np.uint8)

        elif method == 'frequency': # Homomorphic filtering
            if not isinstance(gamma, (float, int)) or gamma <= 0: raise ValueError("Homomorphic gamma must be > 0.")
            if not isinstance(cutoff_low, (float, int)) or not (0 < cutoff_low <= 1.0): raise ValueError("cutoff_low must be in (0, 1].")
            if not isinstance(cutoff_high, (float, int)) or not (cutoff_high >= 1.0): raise ValueError("cutoff_high must be >= 1.0.")
            if cutoff_low >= cutoff_high: raise ValueError("cutoff_low must be less than cutoff_high.")

            if is_color:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_float = l.astype(np.float32) / 255.0 + 1e-5 # Add epsilon before log
                log_l = np.log(l_float)
                fft_l = np.fft.fft2(log_l)
                fft_shift = np.fft.fftshift(fft_l)
            else: # Grayscale
                img_float = image.astype(np.float32) / 255.0 + 1e-5
                log_img = np.log(img_float)
                fft_img = np.fft.fft2(log_img)
                fft_shift = np.fft.fftshift(fft_img)

            rows, cols = image.shape[:2]
            crow, ccol = rows // 2, cols // 2

            # Create Butterworth high-pass filter (inverse of low-pass)
            # Using distance in normalized frequency domain
            u = np.fft.fftfreq(cols) # Freq -0.5 to 0.5
            v = np.fft.fftfreq(rows)
            U, V = np.meshgrid(u, v)
            D = np.sqrt(U**2 + V**2)

            n = 2  # Filter order
            D0 = 0.05 # Cutoff frequency (example, might need tuning)

            # Butterworth High-Pass Filter Transfer Function H(u,v)
            # H = 1 / (1 + (D0 / D)**(2*n)) # Avoid division by zero at D=0
            # Safer way:
            D_clipped = np.maximum(D, 1e-10) # Avoid division by zero
            H = 1.0 / (1.0 + (D0 / D_clipped)**(2*n))

            # Homomorphic filter characteristic: H_homo = (gamma_H - gamma_L) * H + gamma_L
            H_homo = (cutoff_high - cutoff_low) * H + cutoff_low

            # Apply filter
            fft_shift_filtered = fft_shift * H_homo

            # Inverse FFT
            fft_filtered = np.fft.ifftshift(fft_shift_filtered)
            if is_color:
                filtered_log_l = np.fft.ifft2(fft_filtered).real
                exp_l = np.exp(filtered_log_l)
                # Apply gamma correction in spatial domain (different from filter characteristic)
                # Corrected L = exp_l ^ (1/gamma) - This might be double dipping gamma?
                # Let's stick to the standard formulation: apply exp
                corrected_l_float = exp_l # Subtract epsilon added earlier? Maybe not needed after normalization.
                cv2.normalize(corrected_l_float, corrected_l_float, 0, 1, cv2.NORM_MINMAX)
                # Optional: Apply gamma here if needed for contrast adjustment post-filter
                if gamma != 1.0: corrected_l_float = np.power(corrected_l_float, 1.0/gamma)

                corrected_l = (corrected_l_float * 255).astype(np.uint8)
                lab = cv2.merge((corrected_l, a, b))
                corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else: # Grayscale
                filtered_log_img = np.fft.ifft2(fft_filtered).real
                exp_img = np.exp(filtered_log_img)
                corrected_float = exp_img
                cv2.normalize(corrected_float, corrected_float, 0, 1, cv2.NORM_MINMAX)
                if gamma != 1.0: corrected_float = np.power(corrected_float, 1.0/gamma)
                corrected = (corrected_float * 255).astype(np.uint8)

        return corrected

    except cv2.error as e:
        raise ValueError(f"OpenCV error during lighting correction ({method}): {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during lighting correction ({method}): {e}")


# --- Edge Detection ---
def detect_edges(image, method='canny', threshold1=50, threshold2=150,
                 aperture_size=3, L2gradient=False, preset=None):
    """Detect edges using Sobel, Scharr, Laplacian, or Canny. Includes presets for Sobel/Scharr."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")

    # Convert to grayscale if color image
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
         raise ValueError(f"Unsupported image shape for edge detection: {image.shape}")

    # Validate aperture size
    if not isinstance(aperture_size, int) or aperture_size not in [1, 3, 5, 7]: # Size 1 is valid for Scharr
        if method == 'scharr' and aperture_size == -1: # Scharr equivalent uses -1 internally sometimes
             pass # Allow -1 for Scharr maybe? CV2 uses ksize=3 internally though. Let's stick to 3,5,7.
             raise ValueError("aperture_size must be 3, 5, or 7 (except Scharr which implicitly uses 3).")
        elif aperture_size % 2 == 0 or aperture_size <= 0:
             raise ValueError("aperture_size must be an odd positive integer (3, 5, 7).")

    # Validate thresholds
    if not isinstance(threshold1, int) or not (0 <= threshold1 <= 255): raise ValueError("threshold1 must be between 0 and 255.")
    if method == 'canny' and (not isinstance(threshold2, int) or not (0 <= threshold2 <= 255)): raise ValueError("threshold2 must be between 0 and 255 for Canny.")
    if method == 'canny' and threshold1 >= threshold2: raise ValueError("For Canny, threshold1 must be less than threshold2.")


    try:
        edges = None
        if method in ['sobel', 'scharr']:
            # Handle presets
            dx, dy = 1, 1 # Default: detect in both directions
            if preset == 'horizontal': dx, dy = 0, 1
            elif preset == 'vertical': dx, dy = 1, 0

            if method == 'sobel':
                grad_x = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=aperture_size)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=aperture_size)
            else: # Scharr (aperture_size is ignored, uses fixed 3x3)
                 if dx == 1 and dy == 0: # Vertical only
                      grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                      grad_y = np.zeros_like(grad_x) # Need placeholder if only one direction used
                 elif dx == 0 and dy == 1: # Horizontal only
                      grad_x = np.zeros_like(cv2.Scharr(gray, cv2.CV_64F, 0, 1))
                      grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                 else: # Both directions (dx=1, dy=1)
                      grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                      grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)


            # Calculate magnitude
            if L2gradient:
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
            else:
                magnitude = np.abs(grad_x) + np.abs(grad_y)

            # Normalize and threshold
            cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
            magnitude = magnitude.astype(np.uint8)
            _, edges = cv2.threshold(magnitude, threshold1, 255, cv2.THRESH_BINARY)

        elif method == 'laplacian':
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=aperture_size)
            # Take absolute value, normalize, threshold
            magnitude = np.abs(laplacian)
            cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
            magnitude = magnitude.astype(np.uint8)
            _, edges = cv2.threshold(magnitude, threshold1, 255, cv2.THRESH_BINARY)

        elif method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size, L2gradient=L2gradient)

        else:
            raise ValueError(f"Unknown edge detection method: {method}")

        return edges

    except cv2.error as e:
        raise ValueError(f"OpenCV error during edge detection ({method}): {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during edge detection ({method}): {e}")

# --- New Function: Color Space Conversion ---
def convert_color_space(image, target_space='HSV', view_channel=None, **kwargs): # Added **kwargs to ignore unused params from UI potentially
    """Convert image to a target color space and optionally view a single channel."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: image must be a numpy array.")

    # Define available spaces and their corresponding CV2 codes
    # Assumes input is BGR if it's 3-channel
    color_spaces_map = {
        'BGR': None, # No conversion needed if input is BGR
        'RGB': cv2.COLOR_BGR2RGB,
        'GRAY': cv2.COLOR_BGR2GRAY,
        'HSV': cv2.COLOR_BGR2HSV,
        'HLS': cv2.COLOR_BGR2HLS,
        'LAB': cv2.COLOR_BGR2LAB,
        'YCrCb': cv2.COLOR_BGR2YCrCb
    }

    if target_space not in color_spaces_map:
        raise ValueError(f"Invalid target color space: {target_space}")

    try:
        is_color_input = len(image.shape) == 3 and image.shape[2] == 3

        # --- Conversion ---
        conversion_code = color_spaces_map[target_space]
        converted_image = image.copy() # Start with copy

        if conversion_code is not None:
            if not is_color_input and target_space != 'GRAY': # Cannot convert grayscale to most color spaces directly
                 # Can convert grayscale to BGR first
                 if target_space == 'BGR' or target_space == 'RGB':
                      converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                      if target_space == 'RGB':
                           converted_image = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)
                 else:
                      # Maybe convert Gray -> BGR -> Target?
                      # Or raise error? Let's raise error for clarity.
                      raise ValueError(f"Cannot convert grayscale input directly to {target_space}. Convert to BGR first if needed.")
            elif is_color_input:
                 # Ensure input image depth is suitable for the conversion
                 # Some conversions like LAB require CV_8U or CV_32F
                 if image.dtype not in [np.uint8, np.float32]:
                      # Convert to float32 as a safe default if not uint8
                      image_conv_input = image.astype(np.float32)
                      if image.max() > 1.0: # Assume scale 0-255 if max > 1
                           image_conv_input /= 255.0
                 else:
                      image_conv_input = image

                 converted_image = cv2.cvtColor(image_conv_input, conversion_code)

                 # If input was uint8, try converting back if appropriate
                 if image.dtype == np.uint8 and target_space not in ['LAB', 'HSV', 'HLS']: # These often stay float
                     if converted_image.max() <= 1.0 and converted_image.min() >= 0.0: # Check if result is [0,1] range
                          converted_image = (converted_image * 255).astype(np.uint8)

            # If target is GRAY and input is color, conversion happens.
            # If target is GRAY and input is gray, no conversion needed by cvtColor.
            # If target is BGR and input is color, no conversion code needed.

        # --- Channel Viewing ---
        if view_channel is not None:
            if not isinstance(view_channel, int) or view_channel < 0:
                 raise ValueError("view_channel must be a non-negative integer index.")

            # Check if converted image has enough channels
            if len(converted_image.shape) < 3 or converted_image.shape[2] <= view_channel:
                # If target was GRAY, converted_image might be 2D
                if len(converted_image.shape) == 2 and view_channel == 0:
                     # Viewing channel 0 of grayscale is just the grayscale image
                     output_image = converted_image
                else:
                     raise ValueError(f"Cannot view channel {view_channel}. Image shape {converted_image.shape} after conversion to {target_space}.")
            else:
                # Split channels and select the desired one
                channels = cv2.split(converted_image)
                output_image = channels[view_channel]
        else:
             # Return the full converted image
             output_image = converted_image


        # Ensure output is valid numpy array
        if not isinstance(output_image, np.ndarray):
             raise RuntimeError("Color conversion resulted in non-numpy array output.")

        return output_image

    except cv2.error as e:
        raise ValueError(f"OpenCV error during color space conversion to {target_space}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during color space conversion: {e}")


# --- Placeholder/Example for other potential functions ---
# def apply_morphology(image, operation='erode', kernel_shape='rect', kernel_size=3):
#     pass

# def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
#     pass

# def detect_contours(image, mode='RETR_EXTERNAL', method='CHAIN_APPROX_SIMPLE', ...):
#     pass

# def detect_corners(image, method='shi-tomasi', ...):
#     pass