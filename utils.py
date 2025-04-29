"""
Utility functions for DIP-Lib.
Provides functionality for metrics calculation and visualization.
Applying feedback from Gemini, ChatGPT, GROK, and User.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import io
import pandas as pd
import seaborn as sns
import warnings # To suppress specific warnings if needed

def calculate_metrics(original, processed, metrics=None):
    """
    Calculate image quality metrics (PSNR, SSIM) between original and processed images.
    Handles color/grayscale images, potential type/shape mismatches, and calculation errors.

    Args:
        original (numpy.ndarray): Original image.
        processed (numpy.ndarray): Processed image.
        metrics (list, optional): List of metrics to calculate ('psnr', 'ssim').
                                  Defaults to ['psnr', 'ssim'].

    Returns:
        dict: Dictionary containing calculated metric values (e.g., {'psnr': 30.5, 'ssim': 0.95})
              or error/warning messages (e.g., {'error': 'Dimension mismatch'}).
    """
    if metrics is None:
        metrics = ['psnr', 'ssim']

    results = {}

    # --- Input Validation ---
    if not isinstance(original, np.ndarray) or not isinstance(processed, np.ndarray):
        results['error'] = "Invalid input: Both original and processed must be NumPy arrays."
        return results

    if original.size == 0 or processed.size == 0:
         results['error'] = "Invalid input: Image arrays cannot be empty."
         return results

    # --- Ensure images are comparable (Grayscale) ---
    # Metrics like PSNR/SSIM are typically calculated on grayscale intensity.
    try:
        if original.ndim == 3 and original.shape[2] == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        elif original.ndim == 2:
            original_gray = original
        else:
            results['error'] = f"Unsupported original image format (shape: {original.shape}). Must be 2D or 3D (BGR)."
            return results

        if processed.ndim == 3 and processed.shape[2] == 3:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        elif processed.ndim == 2:
            processed_gray = processed
        # Allow single-channel 3D shapes like (H, W, 1) which might come from some operations
        elif processed.ndim == 3 and processed.shape[2] == 1:
             processed_gray = processed.squeeze(axis=2) # Remove the last dimension
        else:
            # Cannot reliably calculate standard metrics against non-standard formats
            results['error'] = f"Unsupported processed image format (shape: {processed.shape}) for standard metrics."
            return results

    except cv2.error as e:
         results['error'] = f"OpenCV error during grayscale conversion: {e}"
         return results
    except Exception as e:
         results['error'] = f"Error during grayscale conversion: {e}"
         return results

    # --- Dimension Matching ---
    if original_gray.shape != processed_gray.shape:
        # Option 1: Resize processed to match original (might affect metrics)
        try:
             processed_gray = cv2.resize(processed_gray, (original_gray.shape[1], original_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
             results['warning'] = "Processed image resized to match original dimensions for metric calculation. This might affect results."
        except cv2.error as e:
             results['error'] = f"Dimension mismatch ({original_gray.shape} vs {processed_gray.shape}) and resize failed: {e}"
             return results
        except Exception as e:
             results['error'] = f"Dimension mismatch ({original_gray.shape} vs {processed_gray.shape}) and resize failed: {e}"
             return results
        # Option 2: Return error (Stricter)
        # results['error'] = f"Image dimensions do not match after grayscale conversion ({original_gray.shape} vs {processed_gray.shape}). Cannot calculate metrics."
        # return results


    # --- Data Type and Range ---
    # Ensure both images have the same numeric dtype for calculations
    try:
        if original_gray.dtype != processed_gray.dtype:
            # Attempt to convert processed to original's dtype
            if np.issubdtype(original_gray.dtype, np.floating) or np.issubdtype(processed_gray.dtype, np.floating):
                 # If either is float, convert both to a common float type (e.g., float64 for precision)
                 original_gray = original_gray.astype(np.float64)
                 processed_gray = processed_gray.astype(np.float64)
            else:
                 # If both are integer types, convert processed to original's type
                 processed_gray = processed_gray.astype(original_gray.dtype)

        # Determine data range for metrics (important!)
        if np.issubdtype(original_gray.dtype, np.integer):
            data_range = np.iinfo(original_gray.dtype).max - np.iinfo(original_gray.dtype).min
        elif np.issubdtype(original_gray.dtype, np.floating):
            # Assume float images are in range [0, 1] unless max > 1
            max_val = max(original_gray.max(), processed_gray.max())
            min_val = min(original_gray.min(), processed_gray.min())
            if max_val <= 1.0 and min_val >= 0.0:
                data_range = 1.0
            elif max_val <= 255.0 and min_val >= 0.0: # Assume range [0, 255] scaled
                 data_range = 255.0
                 # Optionally normalize to [0, 1] here if skimage functions prefer it
                 # original_gray = original_gray / 255.0
                 # processed_gray = processed_gray / 255.0
                 # data_range = 1.0
            else: # Unknown range
                 data_range = max_val - min_val # Use actual range, might be less standard
                 results['warning'] = results.get('warning', '') + " Assuming data range based on image min/max for metrics."
        else:
            results['error'] = f"Unsupported data type for metrics: {original_gray.dtype}"
            return results

    except Exception as e:
        results['error'] = f"Error preparing images for metrics (dtype/range): {e}"
        return results


    # --- Calculate Metrics ---
    # Calculate PSNR
    if 'psnr' in metrics:
        # PSNR requires non-zero data_range and identical shapes
        if data_range > 1e-6 and original_gray.shape == processed_gray.shape:
            try:
                # Suppress RuntimeWarning for potential division by zero if images are identical
                with warnings.catch_warnings():
                     warnings.simplefilter("ignore", category=RuntimeWarning)
                     psnr_val = psnr(original_gray, processed_gray, data_range=data_range)
                results['psnr'] = psnr_val if np.isfinite(psnr_val) else float('inf') # Handle identical images case
            except Exception as e:
                # Catch errors from skimage function
                results['psnr'] = f"Error: {str(e)}"
        else:
            results['psnr'] = "N/A (Zero data range or shape mismatch)"


    # Calculate SSIM
    if 'ssim' in metrics:
        # SSIM requires window size <= image dimensions and odd. Default win_size=7 in skimage >= 0.16
        min_dim = min(original_gray.shape)
        # Default skimage win_size is min(7, min_dim). It must be odd.
        default_win_size = min(7, min_dim)
        if default_win_size % 2 == 0:
             default_win_size -= 1 # Make it odd

        if default_win_size >= 3 and data_range > 1e-6 and original_gray.shape == processed_gray.shape: # Ensure window is valid
             try:
                 # channel_axis=None for 2D images (deprecated in newer skimage, use multichannel=False)
                 # For compatibility, let's try multichannel first
                 try:
                      ssim_val = ssim(original_gray, processed_gray, data_range=data_range,
                                       win_size=default_win_size, multichannel=False) # Explicitly grayscale
                 except TypeError: # Handle older skimage versions without multichannel
                      ssim_val = ssim(original_gray, processed_gray, data_range=data_range,
                                       win_size=default_win_size)

                 results['ssim'] = ssim_val
             except Exception as e:
                 results['ssim'] = f"Error: {str(e)}"
        elif default_win_size < 3:
             results['ssim'] = "N/A (Image too small for default SSIM window)"
        else:
             results['ssim'] = "N/A (Zero data range or shape mismatch)"


    return results


# --- Plotting Functions (Minor or no changes based on feedback) ---

def plot_comparison(images, titles=None, figsize=(12, 8)):
    """Create a comparison grid of images."""
    if not isinstance(images, list) or not images:
         # Handle empty list case
         fig, ax = plt.subplots()
         ax.text(0.5, 0.5, "No images to display.", ha='center', va='center')
         ax.axis('off')
         return fig

    n = len(images)

    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]
    elif len(titles) != n:
         titles = [f"Image {i+1}" for i in range(n)] # Fallback if titles mismatch

    # Adjust layout based on number of images
    ncols = min(n, 4) # Max 4 columns
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False) # squeeze=False ensures axes is always 2D array
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i in range(n):
        image = images[i]
        title = titles[i]

        if not isinstance(image, np.ndarray):
             display_image = None
             effective_title = f"{title}\n(Invalid Image Data)"
        # Convert to RGB if image is BGR
        elif image.ndim == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            effective_title = title
        # Handle grayscale or single channel squeezed
        elif image.ndim == 2:
            display_image = image
            effective_title = title
        else: # Handle other unexpected shapes
             display_image = None # Cannot display reliably
             effective_title = f"{title}\n(Unsupported Shape: {image.shape})"

        if display_image is not None:
             axes[i].imshow(display_image, cmap='gray' if display_image.ndim == 2 else None)
        axes[i].set_title(effective_title)
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def plot_metrics(metrics_data, metric_type='bar'):
    """Create plots for metrics visualization (bar or heatmap)."""
    # Convert metrics data to DataFrame if it's not already
    if isinstance(metrics_data, dict):
        try:
            # Filter out non-numeric values before creating DataFrame for plotting
            plot_data = {k: v for k, v in metrics_data.items() if isinstance(v, (int, float)) and np.isfinite(v)}
            if not plot_data:
                 raise ValueError("No valid numeric metrics found to plot.")
            metrics_df = pd.DataFrame([plot_data]) # Plot as single row
        except Exception as e:
             # Handle error: return empty figure or raise
             fig, ax = plt.subplots()
             ax.text(0.5, 0.5, f"Could not plot metrics:\n{e}", ha='center', va='center')
             ax.axis('off')
             return fig
    elif isinstance(metrics_data, pd.DataFrame):
        # Select only numeric columns for plotting
        metrics_df = metrics_data.select_dtypes(include=np.number)
        if metrics_df.empty:
             fig, ax = plt.subplots()
             ax.text(0.5, 0.5, "No numeric metrics data in DataFrame.", ha='center', va='center')
             ax.axis('off')
             return fig
    else:
        raise ValueError("metrics_data should be a dictionary or DataFrame")

    try:
        if metric_type == 'bar':
            fig, ax = plt.subplots(figsize=(8, 5)) # Adjusted size
            metrics_df.plot(kind='bar', ax=ax, legend=False) # Simple bar for single result usually
            ax.set_ylabel('Value')
            ax.set_title('Metrics')
            ax.tick_params(axis='x', rotation=0) # Horizontal labels if few metrics
            plt.tight_layout()

        elif metric_type == 'heatmap':
            # Heatmap makes more sense if comparing multiple results/methods
            # For single result dict, heatmap is not very informative
            if metrics_df.shape[0] > 1 or metrics_df.shape[1] > 1 : # Check if useful
                 fig, ax = plt.subplots(figsize=(8, 4)) # Adjusted size
                 sns.heatmap(metrics_df, annot=True, cmap='viridis', fmt=".2f", ax=ax)
                 ax.set_title('Metrics Heatmap')
                 plt.tight_layout()
            else: # Fallback to bar plot if heatmap is not suitable
                 fig, ax = plt.subplots(figsize=(8, 5))
                 metrics_df.plot(kind='bar', ax=ax, legend=False)
                 ax.set_ylabel('Value')
                 ax.set_title('Metrics')
                 ax.tick_params(axis='x', rotation=0)
                 plt.tight_layout()


        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

    except Exception as e:
         fig, ax = plt.subplots()
         ax.text(0.5, 0.5, f"Error during plot generation:\n{e}", ha='center', va='center')
         ax.axis('off')
         return fig

    return fig


def plot_histogram(image, bins=256):
    """Plot histogram of an image (BGR or Grayscale)."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Invalid image data for histogram.", ha='center', va='center')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(8, 5)) # Adjusted size

    try:
        if image.ndim == 3 and image.shape[2] == 3:
            # Color image (assume BGR)
            colors = ('b', 'g', 'r')
            legends = ['Blue', 'Green', 'Red']
            for i, color in enumerate(colors):
                # Calculate histogram for channel i
                hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                ax.plot(hist, color=color)
            ax.set_title('Color Histogram')
            ax.legend(legends)
            ax.set_xlim([0, bins]) # Consistent x-axis limit

        elif image.ndim == 2:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            ax.plot(hist, color='gray')
            ax.set_title('Grayscale Histogram')
            ax.set_xlim([0, bins])
        else:
             # Unsupported format
             ax.text(0.5, 0.5, f"Unsupported image shape for histogram:\n{image.shape}", ha='center', va='center')
             ax.axis('off')
             return fig

        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    except cv2.error as e:
         ax.clear() # Clear potential partial plots on error
         ax.text(0.5, 0.5, f"OpenCV Error calculating histogram:\n{e}", ha='center', va='center')
         ax.axis('off')
    except Exception as e:
         ax.clear()
         ax.text(0.5, 0.5, f"Error plotting histogram:\n{e}", ha='center', va='center')
         ax.axis('off')

    return fig

def convert_to_streamlit(fig):
    """Convert matplotlib figure to streamlit compatible format (PNG bytes)."""
    if not fig: return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        buf.seek(0)
        return buf # Return BytesIO object or bytes buf.getvalue()
    except Exception as e:
         # Handle potential errors during savefig
         print(f"Error converting figure to Streamlit format: {e}")
         plt.close(fig) # Ensure figure is closed even on error
         return None