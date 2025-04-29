"""
Utility functions for DIP-Lib.
Provides functionality for metrics calculation and visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import io
import pandas as pd
import seaborn as sns

def calculate_metrics(original, processed, metrics=None):
    """Calculate image quality metrics.
    
    Args:
        original (numpy.ndarray): Original image
        processed (numpy.ndarray): Processed image
        metrics (list): List of metrics to calculate ('psnr', 'ssim')
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    if metrics is None:
        metrics = ['psnr', 'ssim']
    
    results = {}
    
    # Convert images to grayscale if they are color
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # Calculate PSNR
    if 'psnr' in metrics:
        try:
            results['psnr'] = psnr(original_gray, processed_gray)
        except Exception as e:
            results['psnr'] = f"Error: {str(e)}"
    
    # Calculate SSIM
    if 'ssim' in metrics:
        try:
            results['ssim'] = ssim(original_gray, processed_gray)
        except Exception as e:
            results['ssim'] = f"Error: {str(e)}"
    
    return results

def plot_comparison(images, titles=None, figsize=(12, 8)):
    """Create a comparison grid of images.
    
    Args:
        images (list): List of images to compare
        titles (list): List of titles for each image
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure containing the comparison
    """
    n = len(images)
    
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    # Handle the case where only one image is provided
    if n == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        # Convert to RGB if image is BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
            
        axes[i].imshow(display_image, cmap='gray' if len(image.shape) < 3 else None)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_metrics(metrics_data, metric_type='bar'):
    """Create plots for metrics visualization.
    
    Args:
        metrics_data (dict): Metrics data to visualize
        metric_type (str): Type of plot ('bar', 'heatmap')
        
    Returns:
        matplotlib.figure.Figure: Figure containing the plot
    """
    # Convert metrics data to DataFrame if it's not already
    if not isinstance(metrics_data, pd.DataFrame):
        if isinstance(metrics_data, dict):
            # Handle nested dictionaries
            if all(isinstance(v, dict) for v in metrics_data.values()):
                metrics_df = pd.DataFrame(metrics_data).T
            else:
                metrics_df = pd.DataFrame([metrics_data])
        else:
            raise ValueError("metrics_data should be a dictionary or DataFrame")
    else:
        metrics_df = metrics_data
    
    if metric_type == 'bar':
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Value')
        ax.set_title('Metrics Comparison')
        plt.tight_layout()
        
    elif metric_type == 'heatmap':
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(metrics_df, annot=True, cmap='viridis', ax=ax)
        ax.set_title('Metrics Heatmap')
        plt.tight_layout()
        
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    
    return fig

def plot_histogram(image, bins=256):
    """Plot histogram of an image.
    
    Args:
        image (numpy.ndarray): Input image
        bins (int): Number of bins
        
    Returns:
        matplotlib.figure.Figure: Figure containing the histogram
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Color image
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            ax.plot(hist, color=color)
        ax.set_title('Color Histogram')
        ax.legend(['Blue', 'Green', 'Red'])
    else:
        # Grayscale image
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        ax.plot(hist, color='gray')
        ax.set_title('Grayscale Histogram')
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def convert_to_streamlit(fig):
    """Convert matplotlib figure to streamlit compatible format.
    
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure
        
    Returns:
        bytes: Image bytes for Streamlit
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf