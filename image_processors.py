"""
Core image processing functionality for DIP-Lib.
Contains implementations of all six image processing modules.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
import pandas as pd
from utils import calculate_metrics

# Image Sharpening (Unsharp Masking)
def sharpen_image(image, kernel_size=5, weight=1.5):
    """Sharpen an image using Unsharp Masking.

    Args:
        image (numpy.ndarray): Input image
        kernel_size (int): Size of the Gaussian kernel for blurring (must be odd).
        weight (float): Weight of the detail image (original - blurred).

    Returns:
        numpy.ndarray: Sharpened image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Calculate the sharpened image using weighted addition
    # sharpened = original + weight * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + weight, blurred, -weight, 0)

    return sharpened

# Basic Thresholding
def apply_threshold(image, threshold_value=127, threshold_type='binary',
                    use_adaptive=False, adaptive_method='mean', block_size=11, C=2):
    """Apply global or adaptive thresholding to an image.

    Args:
        image (numpy.ndarray): Input image.
        threshold_value (int): Threshold value for global methods.
        threshold_type (str): Type of global thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv').
        use_adaptive (bool): If True, use adaptive thresholding.
        adaptive_method (str): Adaptive method ('mean', 'gaussian').
        block_size (int): Size of the pixel neighborhood area (must be odd).
        C (int): Constant subtracted from the mean or weighted mean.

    Returns:
        numpy.ndarray: Thresholded image (grayscale).
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Map threshold type strings to OpenCV constants
    thresh_types = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV,
        'trunc': cv2.THRESH_TRUNC,
        'tozero': cv2.THRESH_TOZERO,
        'tozero_inv': cv2.THRESH_TOZERO_INV
    }

    # Map adaptive method strings to OpenCV constants
    adaptive_methods = {
        'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
        'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }

    if use_adaptive:
        # Ensure block size is odd and > 1
        if block_size <= 1: block_size = 3
        if block_size % 2 == 0: block_size += 1

        adaptive_thresh_type = thresh_types.get(threshold_type, cv2.THRESH_BINARY) # Adaptive usually uses binary or inv
        adapt_method = adaptive_methods.get(adaptive_method, cv2.ADAPTIVE_THRESH_MEAN_C)

        thresholded = cv2.adaptiveThreshold(gray, 255, adapt_method,
                                            adaptive_thresh_type, block_size, C)
    else:
        thresh_type_flag = thresh_types.get(threshold_type, cv2.THRESH_BINARY)
        _, thresholded = cv2.threshold(gray, threshold_value, 255, thresh_type_flag)

    return thresholded



# Downsampling & Interpolation Analysis
def downsample_interpolate(image, downsample_method='area', scale_factor=0.5, 
                          interpolation_method='bicubic'):
    """Downsample an image and then upsample it using specified methods.
    
    Args:
        image (numpy.ndarray): Input image
        downsample_method (str): Method for downsampling ('simple', 'antialias', 'area')
        scale_factor (float): Scale factor for downsampling
        interpolation_method (str): Method for interpolation ('nearest', 'bilinear', 'bicubic', 'lanczos')
        
    Returns:
        dict: Processed image and metrics
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Map string methods to OpenCV constants
    downsample_methods = {
        'simple': cv2.INTER_NEAREST,
        'antialias': cv2.INTER_AREA,
        'area': cv2.INTER_AREA
    }
    
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    # Ensure valid methods are selected
    if downsample_method not in downsample_methods:
        raise ValueError(f"Invalid downsample method: {downsample_method}")
    if interpolation_method not in interpolation_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation_method}")
    
    # Calculate new dimensions for downsampling
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Downsample the image
    downsampled = cv2.resize(image, (new_w, new_h), 
                             interpolation=downsample_methods[downsample_method])
    
    # Upsample back to original size
    upsampled = cv2.resize(downsampled, (w, h), 
                           interpolation=interpolation_methods[interpolation_method])
    
    # Calculate metrics
    metrics = calculate_metrics(image, upsampled, metrics=['psnr', 'ssim'])
    
    # Return both processed image and metrics
    return {
        'image': upsampled,
        'downsampled': downsampled,  # Include intermediate result
        'metrics': metrics,
        'params': {
            'downsample_method': downsample_method,
            'scale_factor': scale_factor,
            'interpolation_method': interpolation_method
        }
    }

# Geometric Transformations
def geometric_transform(image, rotation=0, scale_x=1.0, scale_y=1.0, 
                        translate_x=0, translate_y=0, shear_x=0, shear_y=0):
    """Apply geometric transformations to an image.
    
    Args:
        image (numpy.ndarray): Input image
        rotation (float): Rotation angle in degrees
        scale_x, scale_y (float): Scaling factors
        translate_x, translate_y (int): Translation in pixels
        shear_x, shear_y (float): Shear factors
        
    Returns:
        numpy.ndarray: Transformed image
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    
    # Apply scaling
    rotation_matrix[0, 0] *= scale_x
    rotation_matrix[1, 1] *= scale_y
    
    # Apply translation
    rotation_matrix[0, 2] += translate_x
    rotation_matrix[1, 2] += translate_y
    
    # Apply shear
    if shear_x != 0 or shear_y != 0:
        shear_matrix = np.array([
            [1, shear_x, 0],
            [shear_y, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Convert rotation matrix to 3x3
        rotation_matrix_3x3 = np.eye(3, dtype=np.float32)
        rotation_matrix_3x3[:2, :3] = rotation_matrix
        
        # Combine matrices
        combined_matrix = np.matmul(rotation_matrix_3x3, shear_matrix)
        
        # Extract the 2x3 part
        transform_matrix = combined_matrix[:2, :3]
    else:
        transform_matrix = rotation_matrix
    
    # Apply the transformation
    result = cv2.warpAffine(image, transform_matrix, (w, h), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return result

# Noise Analysis & Removal
def add_noise(image, noise_type='gaussian', amount=0.05):
    """Add synthetic noise to an image.
    
    Args:
        image (numpy.ndarray): Input image
        noise_type (str): Type of noise ('gaussian', 'salt_pepper')
        amount (float): Noise intensity
        
    Returns:
        numpy.ndarray: Noisy image
    """
    # Convert image to float32 for noise addition
    image_float = image.astype(np.float32) / 255.0
    
    if noise_type == 'gaussian':
        noisy = random_noise(image_float, mode='gaussian', var=amount)
    elif noise_type == 'salt_pepper':
        noisy = random_noise(image_float, mode='s&p', amount=amount)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Convert back to uint8
    noisy_image = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    
    return noisy_image

def remove_noise(image, filter_type='gaussian', params=None):
    """Apply noise removal filter to an image.
    
    Args:
        image (numpy.ndarray): Input image
        filter_type (str): Type of filter ('gaussian', 'median', 'nlm')
        params (dict): Filter-specific parameters
        
    Returns:
        numpy.ndarray: Filtered image
    """
    if params is None:
        params = {}
    
    if filter_type == 'gaussian':
        kernel_size = params.get('kernel_size', 5)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = params.get('sigma', 1.5)
        filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    elif filter_type == 'median':
        kernel_size = params.get('kernel_size', 5)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        filtered = cv2.medianBlur(image, kernel_size)
    
    elif filter_type == 'nlm':
        # For Non-Local Means, we need to handle color and grayscale differently
        h = params.get('h', 10)  # Filter strength
        
        if len(image.shape) == 3:  # Color image
            # Convert to float32
            image_float = image.astype(np.float32) / 255.0
            
            # Estimate noise
            sigma_est = np.mean(estimate_sigma(image_float, multichannel=True))
            
            # Apply NLM
            filtered_float = denoise_nl_means(image_float, h=h*sigma_est, sigma=sigma_est,
                                             fast_mode=True, patch_size=5, patch_distance=7)
            
            # Convert back to uint8
            filtered = np.clip(filtered_float * 255, 0, 255).astype(np.uint8)
        else:  # Grayscale image
            # Convert to float32
            image_float = image.astype(np.float32) / 255.0
            
            # Estimate noise
            sigma_est = estimate_sigma(image_float)
            
            # Apply NLM
            filtered_float = denoise_nl_means(image_float, h=h*sigma_est, sigma=sigma_est,
                                             fast_mode=True, patch_size=5, patch_distance=7)
            
            # Convert back to uint8
            filtered = np.clip(filtered_float * 255, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return filtered

# Image Enhancement
def enhance_image(image, gamma=1.0, use_clahe=False, clip_limit=2.0, 
                 tile_grid_size=(8, 8), use_equalization=False):
    """Enhance an image using various techniques.
    
    Args:
        image (numpy.ndarray): Input image
        gamma (float): Gamma correction value
        use_clahe (bool): Whether to use CLAHE
        clip_limit (float): CLAHE clip limit
        tile_grid_size (tuple): CLAHE tile grid size
        use_equalization (bool): Whether to use histogram equalization
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    enhanced = image.copy()
    
    # Apply gamma correction
    if gamma != 1.0:
        # Create lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)
    
    # Handle color vs grayscale for histogram operations
    if len(enhanced.shape) == 3:  # Color image
        # Convert to LAB color space for CLAHE (only on L channel)
        if use_clahe:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply histogram equalization to each channel
        if use_equalization:
            # Split channels
            b, g, r = cv2.split(enhanced)
            
            # Apply histogram equalization to each channel
            b = cv2.equalizeHist(b)
            g = cv2.equalizeHist(g)
            r = cv2.equalizeHist(r)
            
            # Merge channels
            enhanced = cv2.merge((b, g, r))
    
    else:  # Grayscale image
        # Apply CLAHE
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(enhanced)
        
        # Apply histogram equalization
        if use_equalization:
            enhanced = cv2.equalizeHist(enhanced)
    
    return enhanced



# Lighting Correction
def correct_lighting(image, method='spatial', kernel_size=51, gamma=1.5, 
                    cutoff_low=0.5, cutoff_high=2.0):
    """Correct uneven lighting in an image.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Correction method ('spatial', 'frequency')
        kernel_size (int): Kernel size for spatial method
        gamma (float): Gamma for homomorphic filtering
        cutoff_low, cutoff_high (float): Cutoff frequencies
        
    Returns:
        numpy.ndarray: Corrected image
    """
    # Convert to float32 for processing
    image_float = image.astype(np.float32) / 255.0
    
    if method == 'spatial':
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if len(image.shape) == 3:  # Color image
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply lighting correction to L channel
            l_float = l.astype(np.float32) / 255.0
            
            # Estimate background illumination using large Gaussian blur
            background = cv2.GaussianBlur(l_float, (kernel_size, kernel_size), 0)
            
            # Divide by background to remove illumination variation
            corrected_l = l_float / (background + 0.01)  # Add small value to avoid division by zero
            
            # Normalize to [0, 1]
            corrected_l = (corrected_l - np.min(corrected_l)) / (np.max(corrected_l) - np.min(corrected_l))
            
            # Convert back to 8-bit and merge channels
            corrected_l = (corrected_l * 255).astype(np.uint8)
            lab = cv2.merge((corrected_l, a, b))
            
            # Convert back to BGR
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        else:  # Grayscale image
            # Estimate background illumination using large Gaussian blur
            background = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 0)
            
            # Divide by background to remove illumination variation
            corrected_float = image_float / (background + 0.01)  # Add small value to avoid division by zero
            
            # Normalize to [0, 1]
            corrected_float = (corrected_float - np.min(corrected_float)) / (np.max(corrected_float) - np.min(corrected_float))
            
            # Convert back to 8-bit
            corrected = (corrected_float * 255).astype(np.uint8)
    
    elif method == 'frequency':
        # Implementation of homomorphic filtering
        if len(image.shape) == 3:  # Color image
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Process L channel
            l_float = l.astype(np.float32) / 255.0
            
            # Take log transform
            log_l = np.log1p(l_float)  # log(1+x) to avoid log(0)
            
            # Apply FFT
            fft_l = np.fft.fft2(log_l)
            fft_shift = np.fft.fftshift(fft_l)
            
            # Create high-pass filter
            rows, cols = l.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create a butterworth high-pass filter
            x = np.linspace(-0.5, 0.5, cols)
            y = np.linspace(-0.5, 0.5, rows)
            u, v = np.meshgrid(x, y)
            d = np.sqrt(u*u + v*v)
            
            # Butterworth filter
            n = 2  # Filter order
            d0 = 0.05  # Cutoff frequency
            
            hp_filter = 1.0 - 1.0 / (1.0 + (d / d0) ** (2 * n))
            
            # Adjust filter for homomorphic filtering
            hp_filter = (cutoff_high - cutoff_low) * hp_filter + cutoff_low
            
            # Apply filter to frequency domain
            fft_shift_filtered = fft_shift * hp_filter
            
            # Inverse FFT
            fft_filtered = np.fft.ifftshift(fft_shift_filtered)
            filtered_l = np.fft.ifft2(fft_filtered).real
            
            # Apply exponential to reverse log transform
            corrected_l = np.expm1(filtered_l)  # exp(x)-1 to reverse log1p
            
            # Normalize to [0, 1]
            corrected_l = (corrected_l - np.min(corrected_l)) / (np.max(corrected_l) - np.min(corrected_l))
            
            # Apply gamma correction for additional enhancement
            corrected_l = np.power(corrected_l, 1/gamma)
            
            # Convert back to 8-bit and merge channels
            corrected_l = (corrected_l * 255).astype(np.uint8)
            lab = cv2.merge((corrected_l, a, b))
            
            # Convert back to BGR
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        else:  # Grayscale image
            # Take log transform
            log_img = np.log1p(image_float)  # log(1+x) to avoid log(0)
            
            # Apply FFT
            fft_img = np.fft.fft2(log_img)
            fft_shift = np.fft.fftshift(fft_img)
            
            # Create high-pass filter
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create a butterworth high-pass filter
            x = np.linspace(-0.5, 0.5, cols)
            y = np.linspace(-0.5, 0.5, rows)
            u, v = np.meshgrid(x, y)
            d = np.sqrt(u*u + v*v)
            
            # Butterworth filter
            n = 2  # Filter order
            d0 = 0.05  # Cutoff frequency
            
            hp_filter = 1.0 - 1.0 / (1.0 + (d / d0) ** (2 * n))
            
            # Adjust filter for homomorphic filtering
            hp_filter = (cutoff_high - cutoff_low) * hp_filter + cutoff_low
            
            # Apply filter to frequency domain
            fft_shift_filtered = fft_shift * hp_filter
            
            # Inverse FFT
            fft_filtered = np.fft.ifftshift(fft_shift_filtered)
            filtered_img = np.fft.ifft2(fft_filtered).real
            
            # Apply exponential to reverse log transform
            corrected_float = np.expm1(filtered_img)  # exp(x)-1 to reverse log1p
            
            # Normalize to [0, 1]
            corrected_float = (corrected_float - np.min(corrected_float)) / (np.max(corrected_float) - np.min(corrected_float))
            
            # Apply gamma correction for additional enhancement
            corrected_float = np.power(corrected_float, 1/gamma)
            
            # Convert back to 8-bit
            corrected = (corrected_float * 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown lighting correction method: {method}")
    
    return corrected

# Edge Detection
def detect_edges(image, method='sobel', threshold1=100, threshold2=200, 
                aperture_size=3, L2gradient=False):
    """Detect edges in an image.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Edge detection method ('sobel', 'scharr', 'laplacian', 'canny')
        threshold1, threshold2 (int): Thresholds for Canny
        aperture_size (int): Aperture size
        L2gradient (bool): Use L2 gradient
        
    Returns:
        numpy.ndarray: Edge map
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'sobel':
        # Apply Sobel in x and y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_size)
        
        # Calculate gradient magnitude
        if L2gradient:
            # L2 norm (Euclidean distance)
            edges = np.sqrt(sobelx**2 + sobely**2)
        else:
            # L1 norm (Manhattan distance)
            edges = np.abs(sobelx) + np.abs(sobely)
        
        # Normalize to 0-255 range
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype(np.uint8)
        
        # Apply threshold
        _, edges = cv2.threshold(edges, threshold1, 255, cv2.THRESH_BINARY)
    
    elif method == 'scharr':
        # Apply Scharr in x and y directions
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        
        # Calculate gradient magnitude
        if L2gradient:
            # L2 norm (Euclidean distance)
            edges = np.sqrt(scharrx**2 + scharry**2)
        else:
            # L1 norm (Manhattan distance)
            edges = np.abs(scharrx) + np.abs(scharry)
        
        # Normalize to 0-255 range
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype(np.uint8)
        
        # Apply threshold
        _, edges = cv2.threshold(edges, threshold1, 255, cv2.THRESH_BINARY)
    
    elif method == 'laplacian':
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=aperture_size)
        
        # Take absolute value
        edges = np.abs(laplacian)
        
        # Normalize to 0-255 range
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype(np.uint8)
        
        # Apply threshold
        _, edges = cv2.threshold(edges, threshold1, 255, cv2.THRESH_BINARY)
    
    elif method == 'canny':
        # Apply Canny edge detector
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size, L2gradient=L2gradient)
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    return edges