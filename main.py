"""
DIP-Lib: Digital Image Processing Library
Main Streamlit application.

Applying feedback from Gemini, ChatGPT, GROK, and User.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import io
import time # For progress simulation/spinner
import zipfile # For downloading multiple images

# Import our modules
import image_processors as ip
from utils import plot_comparison, plot_metrics, plot_histogram, convert_to_streamlit, calculate_metrics # Ensure calculate_metrics is imported

# --- Constants for Defaults ---
DEFAULT_PARAMS = {
    "Downsampling & Interpolation": {'downsample_method': 'area', 'scale_factor': 0.5, 'interpolation_method': 'bicubic'},
    "Geometric Transformations": {'rotation': 0, 'scale_x': 1.0, 'scale_y': 1.0, 'translate_x': 0, 'translate_y': 0, 'shear_x': 0.0, 'shear_y': 0.0},
    # Noise module split into Add Noise and Remove Noise
    "Add Noise": {'noise_type': 'gaussian', 'amount': 0.05},
    "Remove Noise": {'filter_type': 'median', 'kernel_size': 5, 'sigma': 1.5, 'h': 10}, # Include defaults for all possible params
    "Image Enhancement": {'gamma': 1.0, 'use_clahe': False, 'clip_limit': 2.0, 'tile_grid_size': (8, 8), 'use_equalization': False},
    "Lighting Correction": {'method': 'spatial', 'kernel_size': 51, 'gamma': 1.5, 'cutoff_low': 0.5, 'cutoff_high': 2.0},
    "Edge Detection": {'method': 'canny', 'threshold1': 50, 'threshold2': 150, 'aperture_size': 3, 'L2gradient': False, 'preset': 'canny_default'}, # Default threshold changed slightly for range slider
    "Sharpening": {'kernel_size': 5, 'weight': 1.5},
    "Thresholding": {'threshold_value': 127, 'threshold_type': 'binary', 'use_adaptive': False, 'adaptive_method': 'mean', 'block_size': 11, 'C': 2},
    "Color Space Conversion": {'target_space': 'HSV', 'view_channel': None}, # New Module
    # Add other new modules if implemented
    # "Morphological Ops": {},
    # "Bilateral Filter": {},
    # "Contour Detection": {},
    # "Corner Detection": {},
}

AVAILABLE_MODULES = list(DEFAULT_PARAMS.keys())


# --- Helper Functions ---

def get_module_defaults(module_name):
    """Returns a copy of default parameters for a module."""
    return DEFAULT_PARAMS.get(module_name, {}).copy()

# Consider caching image loading
# @st.cache_data # Caching can speed up re-runs if the same file is uploaded
def load_image(uploaded_file):
    """Load an image from uploaded file with error handling."""
    if uploaded_file is None:
        return None, "Please upload an image."

    try:
        # Read the file
        image = Image.open(uploaded_file)

        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Handle RGBA -> RGB/BGR (common issue with PNGs)
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
             st.warning("Input image has an Alpha channel. Converting to RGB before processing.")
             image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB) # Convert to RGB first

        # Convert RGB to BGR (OpenCV format)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Check if it's already BGR (less common from PIL but possible)
            # This check is heuristic and might not be perfect
            is_rgb = True # Assume RGB by default from PIL
            # Add checks if needed, e.g., if channels seem swapped
            if is_rgb:
                 image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                 image_array_bgr = image_array # Assume it was already BGR

        elif len(image_array.shape) == 2: # Grayscale
            image_array_bgr = image_array # Keep as is
        else:
            return None, f"Unsupported image format/shape: {image_array.shape}. Please use RGB or Grayscale."

        return image_array_bgr, None # Return image and no error
    except Exception as e:
        # More specific error handling can be added (e.g., for file read errors)
        error_msg = f"Image loading failed. Ensure it’s a valid JPG, JPEG, or PNG file. Error: {e}"
        st.error(error_msg)
        return None, error_msg


def build_module_ui(module_name):
    """Build the UI components for a specific module with enhancements."""
    st.subheader(f"{module_name} Settings")

    # Initialize params with defaults for the specific module
    # Use session state to remember parameters within a module interaction
    if f"{module_name}_params" not in st.session_state:
        st.session_state[f"{module_name}_params"] = get_module_defaults(module_name)

    params = st.session_state[f"{module_name}_params"]

    # --- Reset Button ---
    if st.button(f"Reset '{module_name}' Parameters", key=f"reset_{module_name}"):
        params = get_module_defaults(module_name)
        st.session_state[f"{module_name}_params"] = params
        st.rerun() # Rerun to reflect reset values in widgets

    # --- UI Elements per module ---
    if module_name == "Downsampling & Interpolation":
        col1, col2 = st.columns(2)
        with col1:
            params['downsample_method'] = st.selectbox(
                "Downsampling Method",
                options=["simple", "antialias", "area"], index=["simple", "antialias", "area"].index(params.get('downsample_method','area')),
                key=f"{module_name}_downsample_method",
                 help="Method for shrinking the image. 'area' is generally best for downsampling."
            )
            params['scale_factor'] = st.slider(
                "Scale Factor", min_value=0.1, max_value=0.9, value=params.get('scale_factor', 0.5), step=0.1,
                key=f"{module_name}_scale_factor",
                help="Factor by which to shrink the image (e.g., 0.5 means half size)."
            )
        with col2:
            params['interpolation_method'] = st.selectbox(
                "Interpolation Method",
                options=["nearest", "bilinear", "bicubic", "lanczos"], index=["nearest", "bilinear", "bicubic", "lanczos"].index(params.get('interpolation_method','bicubic')),
                key=f"{module_name}_interpolation_method",
                help="Method for resizing the image back up. 'bicubic' or 'lanczos' often give better quality."
            )

    elif module_name == "Geometric Transformations":
         # Use expander for less common parameters
         with st.expander("Basic Transformations", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                params['rotation'] = st.slider("Rotation (degrees)", -180, 180, params.get('rotation',0), key=f"{module_name}_rotation", help="Rotate the image around its center.")
            with col2:
                params['scale_x'] = st.slider("Scale X", 0.1, 2.0, params.get('scale_x',1.0), 0.1, key=f"{module_name}_scale_x", help="Stretch/shrink horizontally.")
                params['scale_y'] = st.slider("Scale Y", 0.1, 2.0, params.get('scale_y',1.0), 0.1, key=f"{module_name}_scale_y", help="Stretch/shrink vertically.")

         with st.expander("Advanced Transformations"):
             col3, col4 = st.columns(2)
             with col3:
                 params['translate_x'] = st.slider("Translate X (pixels)", -100, 100, params.get('translate_x',0), key=f"{module_name}_translate_x", help="Shift image left/right.")
                 params['translate_y'] = st.slider("Translate Y (pixels)", -100, 100, params.get('translate_y',0), key=f"{module_name}_translate_y", help="Shift image up/down.")
             with col4:
                 params['shear_x'] = st.slider("Shear X", -1.0, 1.0, params.get('shear_x',0.0), 0.1, key=f"{module_name}_shear_x", help="Slant the image horizontally.")
                 params['shear_y'] = st.slider("Shear Y", -1.0, 1.0, params.get('shear_y',0.0), 0.1, key=f"{module_name}_shear_y", help="Slant the image vertically.")

    # --- Split Noise Module ---
    elif module_name == "Add Noise":
         col1, col2 = st.columns(2)
         with col1:
            params['noise_type'] = st.selectbox(
                "Noise Type", options=["gaussian", "salt_pepper"], index=["gaussian", "salt_pepper"].index(params.get('noise_type','gaussian')),
                key=f"{module_name}_noise_type", help="Type of synthetic noise to add."
            )
         with col2:
             params['amount'] = st.slider(
                 "Noise Amount/Variance", min_value=0.01, max_value=0.2, value=params.get('amount',0.05), step=0.01,
                 key=f"{module_name}_noise_amount", help="Controls intensity (variance for Gaussian, density for S&P)."
             )

    elif module_name == "Remove Noise":
         col1, col2 = st.columns(2)
         with col1:
             params['filter_type'] = st.selectbox(
                 "Filter Type", options=["gaussian", "median", "nlm"], index=["gaussian", "median", "nlm"].index(params.get('filter_type','median')),
                 key=f"{module_name}_filter_type",
                 help="Algorithm to remove noise. 'Median' good for S&P, 'Gaussian' for general blur, 'NLM' powerful but slower."
             )
         with col2:
             if params['filter_type'] in ['gaussian', 'median']:
                 params['kernel_size'] = st.slider(
                     "Kernel Size", min_value=3, max_value=15, value=params.get('kernel_size',5), step=2,
                     key=f"{module_name}_kernel_size", help="Size of the filter window (must be odd)."
                 )
                 if params['filter_type'] == 'gaussian':
                     params['sigma'] = st.slider(
                         "Sigma (Gaussian Blur)", min_value=0.1, max_value=5.0, value=params.get('sigma',1.5), step=0.1,
                         key=f"{module_name}_sigma", help="Standard deviation of the Gaussian kernel (controls blur amount)."
                     )
             elif params['filter_type'] == 'nlm':
                 params['h'] = st.slider(
                     "Filter Strength (h) for NLM", min_value=1, max_value=20, value=params.get('h',10), step=1,
                     key=f"{module_name}_h", help="Controls how aggressively NLM denoises. Higher values remove more noise but may blur details."
                 )

    elif module_name == "Image Enhancement":
         col1, col2 = st.columns(2)
         with col1:
             params['gamma'] = st.slider("Gamma", 0.1, 3.0, params.get('gamma',1.0), 0.1, key=f"{module_name}_gamma", help="Adjusts image brightness. <1=darker, >1=brighter.")
             params['use_equalization'] = st.checkbox("Use Histogram Equalization", value=params.get('use_equalization',False), key=f"{module_name}_use_equalization", help="Stretches intensity range to improve contrast globally.")
         with col2:
             params['use_clahe'] = st.checkbox("Use CLAHE", value=params.get('use_clahe',False), key=f"{module_name}_use_clahe", help="Adaptive histogram equalization for better local contrast.")
             if params['use_clahe']:
                 params['clip_limit'] = st.slider("CLAHE Clip Limit", 0.5, 10.0, params.get('clip_limit',2.0), 0.5, key=f"{module_name}_clip_limit", help="Threshold for contrast limiting in CLAHE.")
                 tile_size_val = params.get('tile_grid_size', (8, 8))[0] # Get current size from tuple
                 tile_size = st.slider("CLAHE Tile Size", 2, 16, tile_size_val, 2, key=f"{module_name}_tile_grid_size_slider", help="Size of local regions for applying CLAHE.")
                 params['tile_grid_size'] = (tile_size, tile_size) # Store as tuple

    elif module_name == "Lighting Correction":
        params['method'] = st.radio(
            "Correction Method", options=["spatial", "frequency"], index=["spatial", "frequency"].index(params.get('method','spatial')), horizontal=True,
            key=f"{module_name}_method", help="'Spatial' uses blurring, 'Frequency' uses Homomorphic filtering (often better for complex lighting)."
        )
        if params['method'] == 'spatial':
             params['kernel_size'] = st.slider(
                 "Background Kernel Size", min_value=11, max_value=201, value=params.get('kernel_size',51), step=10,
                 key=f"{module_name}_kernel_size_spatial", help="Size of Gaussian kernel to estimate background illumination (must be odd, large)."
             )
        elif params['method'] == 'frequency':
             with st.expander("Homomorphic Filter Parameters", expanded=True):
                 col1, col2 = st.columns(2)
                 with col1:
                     params['gamma'] = st.slider("Gamma (Homomorphic)", 0.1, 3.0, params.get('gamma',1.5), 0.1, key=f"{module_name}_gamma_freq", help="Controls intensity range compression/expansion.")
                     params['cutoff_low'] = st.slider("Low Frequency Gain", 0.1, 1.0, params.get('cutoff_low',0.5), 0.1, key=f"{module_name}_cutoff_low", help="Gain for low frequencies (illumination). Lower values decrease illumination component more.")
                 with col2:
                     params['cutoff_high'] = st.slider("High Frequency Gain", 1.0, 5.0, params.get('cutoff_high',2.0), 0.1, key=f"{module_name}_cutoff_high", help="Gain for high frequencies (reflectance/details). Higher values enhance details more.")


    elif module_name == "Edge Detection":
         col1, col2 = st.columns(2)
         with col1:
             params['method'] = st.selectbox(
                 "Method", options=["sobel", "scharr", "laplacian", "canny"], index=["sobel", "scharr", "laplacian", "canny"].index(params.get('method','canny')),
                 key=f"{module_name}_method", help="Edge detection algorithm. Canny is often preferred."
             )

             # --- Presets for Sobel/Scharr ---
             if params['method'] in ['sobel', 'scharr']:
                 preset_options = {'default': "Default (Both Directions)", 'horizontal': "Horizontal Edges", 'vertical': "Vertical Edges"}
                 selected_preset = st.selectbox("Direction Preset", options=list(preset_options.keys()), format_func=lambda x: preset_options[x], index=0, key=f"{module_name}_preset")
                 # Apply preset - Note: Presets would ideally set dx, dy, ksize etc. in image_processors.py
                 # For now, just store the preset choice. Actual logic needs processor update.
                 params['preset'] = selected_preset
                 st.caption("(Preset logic needs update in image_processors.py)") # Reminder

             # --- Parameter Controls ---
             if params['method'] == 'canny':
                 # Use range slider for Canny thresholds
                 low_thresh = params.get('threshold1', 50)
                 high_thresh = params.get('threshold2', 150)
                 # Ensure low < high before setting slider value
                 if low_thresh >= high_thresh: low_thresh = high_thresh - 10

                 canny_thresholds = st.slider(
                     "Canny Thresholds (Low, High)", min_value=0, max_value=255, value=(low_thresh, high_thresh),
                     key=f"{module_name}_canny_range", help="Lower/Upper thresholds for hysteresis linking. Enforces Low < High."
                 )
                 params['threshold1'] = canny_thresholds[0]
                 params['threshold2'] = canny_thresholds[1]
             elif params['method'] != 'canny':
                 params['threshold1'] = st.slider(
                     "Threshold", min_value=0, max_value=255, value=params.get('threshold1',100), step=5,
                     key=f"{module_name}_threshold1", help="Threshold for Sobel/Scharr/Laplacian output."
                 )

         with col2:
             params['aperture_size'] = st.select_slider(
                 "Aperture Size", options=[3, 5, 7], value=params.get('aperture_size',3),
                 key=f"{module_name}_aperture_size", help="Size of the kernel for Sobel/Scharr/Laplacian/Canny (must be odd)."
             )
             params['L2gradient'] = st.checkbox("Use L2 Gradient (Canny/Sobel/Scharr)", value=params.get('L2gradient',False), key=f"{module_name}_L2gradient", help="Use more accurate Euclidean distance for gradient magnitude.")

         st.info("Tip: Consider applying 'Remove Noise' (e.g., Gaussian Blur) before Edge Detection for cleaner results.")


    elif module_name == "Sharpening":
         col1, col2 = st.columns(2)
         with col1:
             params['kernel_size'] = st.slider(
                 "Blur Kernel Size", min_value=3, max_value=25, value=params.get('kernel_size',5), step=2,
                 key=f"{module_name}_kernel_size", help="Size of Gaussian kernel for blurring in Unsharp Masking (must be odd)."
             )
         with col2:
             params['weight'] = st.slider(
                 "Sharpening Weight", min_value=0.1, max_value=5.0, value=params.get('weight',1.5), step=0.1,
                 key=f"{module_name}_weight", help="How strongly to add the sharpened details back."
             )

    elif module_name == "Thresholding":
        params['use_adaptive'] = st.checkbox("Use Adaptive Thresholding", value=params.get('use_adaptive',False), key=f"{module_name}_adaptive_thresh_check", help="Calculate threshold per region, good for uneven lighting.")

        col1, col2 = st.columns(2)
        with col1:
            if params['use_adaptive']:
                params['adaptive_method'] = st.selectbox(
                    "Adaptive Method", options=['mean', 'gaussian'], index=['mean', 'gaussian'].index(params.get('adaptive_method','mean')),
                    key=f"{module_name}_adaptive_method", help="'Mean': threshold is mean of neighborhood. 'Gaussian': threshold is weighted sum."
                )
                params['block_size'] = st.slider(
                    "Block Size", min_value=3, max_value=51, value=params.get('block_size',11), step=2,
                    key=f"{module_name}_block_size", help="Size of the pixel neighborhood area (must be odd)."
                )
            else:
                 params['threshold_value'] = st.slider(
                     "Global Threshold Value", min_value=0, max_value=255, value=params.get('threshold_value',127),
                     key=f"{module_name}_threshold_value", help="Pixels below this value become 0, above become 255 (for binary type)."
                 )
                 # Suggestion for Otsu/Triangle (requires processor update)
                 st.caption("[Future Idea: Add Otsu/Triangle buttons for automatic threshold]")


        with col2:
             if params['use_adaptive']:
                 # Only show valid options for adaptive
                 thresh_options = ['binary', 'binary_inv']
                 default_thresh_type = params.get('threshold_type', 'binary')
                 current_index = thresh_options.index(default_thresh_type) if default_thresh_type in thresh_options else 0
                 params['threshold_type'] = st.selectbox(
                     "Threshold Type (Adaptive)", options=thresh_options, index=current_index,
                     key=f"{module_name}_threshold_type_adaptive", help="How to apply the threshold."
                 )
                 params['C'] = st.slider(
                     "Constant C", min_value=-20, max_value=20, value=params.get('C',2),
                     key=f"{module_name}_C", help="Constant subtracted from the calculated mean/weighted mean."
                 )
             else:
                 # Show all options for global thresholding
                 thresh_options = ['binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
                 default_thresh_type = params.get('threshold_type', 'binary')
                 current_index = thresh_options.index(default_thresh_type) if default_thresh_type in thresh_options else 0
                 params['threshold_type'] = st.selectbox(
                     "Threshold Type (Global)", options=thresh_options, index=current_index,
                     key=f"{module_name}_threshold_type_global", help="Different ways to apply the global threshold."
                 )


    # --- New Module: Color Space Conversion ---
    elif module_name == "Color Space Conversion":
         col1, col2 = st.columns(2)
         with col1:
             # Define available spaces and their corresponding CV2 codes (if needed)
             color_spaces = {
                'BGR': None, # Original (usually)
                'RGB': cv2.COLOR_BGR2RGB,
                'GRAY': cv2.COLOR_BGR2GRAY,
                'HSV': cv2.COLOR_BGR2HSV,
                'HLS': cv2.COLOR_BGR2HLS,
                'LAB': cv2.COLOR_BGR2LAB,
                'YCrCb': cv2.COLOR_BGR2YCrCb
             }
             params['target_space'] = st.selectbox(
                 "Target Color Space", options=list(color_spaces.keys()), index=list(color_spaces.keys()).index(params.get('target_space','HSV')),
                 key=f"{module_name}_target_space", help="Convert image to a different color representation."
             )
         with col2:
             target = params['target_space']
             # Option to view single channel (only if target space is multi-channel)
             if target not in ['GRAY', 'BGR', 'RGB']: # BGR/RGB channels are less common to view individually here
                 channel_options = {
                     'HSV': ['None', 'H (0)', 'S (1)', 'V (2)'],
                     'HLS': ['None', 'H (0)', 'L (1)', 'S (2)'],
                     'LAB': ['None', 'L* (0)', 'a* (1)', 'b* (2)'],
                     'YCrCb': ['None', 'Y (0)', 'Cr (1)', 'Cb (2)']
                 }
                 valid_options = channel_options.get(target, ['None'])
                 # Ensure default is valid
                 default_channel_label = params.get('view_channel_label', 'None')
                 if default_channel_label not in valid_options: default_channel_label = 'None'

                 view_channel_label = st.selectbox(
                     "View Single Channel (Grayscale)", options=valid_options, index=valid_options.index(default_channel_label),
                     key=f"{module_name}_view_channel_label", help="Show only one channel of the converted space."
                 )
                 params['view_channel_label'] = view_channel_label # Store the label

                 # Map label back to channel index or None
                 if view_channel_label == 'None':
                     params['view_channel'] = None
                 else:
                     try:
                         params['view_channel'] = int(view_channel_label.split('(')[1].split(')')[0])
                     except:
                         params['view_channel'] = None # Fallback
             else:
                 params['view_channel'] = None # Cannot select channel for Gray/BGR/RGB easily here
                 params['view_channel_label'] = 'None'


    # --- Placeholder for other new modules ---
    # elif module_name == "Morphological Ops": st.warning("Morphological Ops UI not implemented yet.")
    # elif module_name == "Bilateral Filter": st.warning("Bilateral Filter UI not implemented yet.")
    # elif module_name == "Contour Detection": st.warning("Contour Detection UI not implemented yet.")
    # elif module_name == "Corner Detection": st.warning("Corner Detection UI not implemented yet.")


    # Store updated params back into session state
    st.session_state[f"{module_name}_params"] = params
    return params


# Consider caching processing steps if inputs/params are identical
# @st.cache_data # Apply carefully, might need complex hashing for image data + params
def apply_pipeline_step(input_image, func, params, module_name):
    """Applies a single pipeline step with error handling."""
    if input_image is None:
        st.warning(f"Skipping step '{module_name}': No valid input image.")
        return None, "Skipped: No valid input."

    # Ensure input is numpy array
    if not isinstance(input_image, np.ndarray):
         st.warning(f"Skipping step '{module_name}': Input is not a valid image array.")
         return None, "Skipped: Invalid input type."


    try:
        if module_name == "Downsampling & Interpolation":
            result = func(input_image, **params)
            output_image = result['image']
            step_info = f"{module_name} (DS: {params['downsample_method']}, Scale: {params['scale_factor']}, Interp: {params['interpolation_method']})"
        elif module_name == "Add Noise":
             output_image = func(input_image, **params)
             step_info = f"{module_name} (Type: {params['noise_type']}, Amount: {params['amount']})"
        elif module_name == "Remove Noise":
             output_image = func(input_image, **params)
             step_info = f"{module_name} (Filter: {params['filter_type']})" # Add more param details if needed
        elif module_name == "Color Space Conversion":
             # NOTE: Requires `ip.convert_color_space` function in image_processors.py
             if hasattr(ip, 'convert_color_space'):
                  output_image = ip.convert_color_space(input_image, **params)
                  step_info = f"{module_name} (To: {params['target_space']}, Channel: {params.get('view_channel_label','All')})"
             else:
                  st.error(f"Missing function 'ip.convert_color_space' for module '{module_name}'. Skipping.")
                  output_image = input_image # Pass original image through
                  step_info = f"{module_name} - Error: Function Missing!"
        else:
            # Default handling for other modules
            output_image = func(input_image, **params)
            step_info = f"{module_name}" # Basic name, can be customized

        # Basic validation of output
        if not isinstance(output_image, np.ndarray):
            raise ValueError("Processing function did not return a numpy array.")

        return output_image, step_info

    except Exception as e:
        st.error(f"Error during '{module_name}' processing: {e}")
        # Decide how to handle error: return original image or None
        return input_image, f"{module_name} - Error!" # Return input image to allow pipeline continuation


def apply_pipeline(initial_image, pipeline_steps):
    """Apply a sequence of processing steps to an image with progress."""
    current_image = initial_image.copy()
    intermediate_results = [("Original", initial_image)] # Store tuples (step_name, image_array)
    step_info_list = ["Original"]

    if not pipeline_steps:
        return initial_image, intermediate_results, step_info_list

    # Progress bar/spinner
    with st.spinner(f'Applying {len(pipeline_steps)} pipeline step(s)...'):
        start_time = time.time()

        for i, (func, params, module_name) in enumerate(pipeline_steps):
             step_start_time = time.time()
             st.write(f"... Applying Step {i+1}: {module_name}") # Optional: show progress text

             # --- Apply step using helper function ---
             processed_image, step_info = apply_pipeline_step(current_image, func, params, module_name)

             if processed_image is not None:
                 current_image = processed_image
                 intermediate_results.append((f"Step {i+1}: {step_info}", current_image))
                 step_info_list.append(step_info)
             else:
                 # Handle step failure if needed (e.g., stop pipeline)
                 st.warning(f"Step {i+1} ('{module_name}') failed or was skipped. Continuing with previous result.")
                 intermediate_results.append((f"Step {i+1}: {step_info}", current_image)) # Add placeholder
                 step_info_list.append(step_info)
                 # Optionally: break # Stop pipeline on first error


             step_end_time = time.time()
             # print(f"Step {i+1} ({module_name}) took {step_end_time - step_start_time:.2f}s") # Debug timing

        end_time = time.time()
        st.success(f"Pipeline executed in {end_time - start_time:.2f} seconds.")

    return current_image, intermediate_results, step_info_list


def save_image_to_bytes(image, format="PNG"):
     """Saves a single numpy image to bytes buffer."""
     if image is None: return None
     try:
         # Convert BGR to RGB for saving with PIL if it's color
         if len(image.shape) == 3 and image.shape[2] == 3:
             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         elif len(image.shape) == 2: # Grayscale
             image_rgb = image # PIL handles grayscale
         elif len(image.shape) == 3 and image.shape[2] == 1: # Single channel e.g. from split
              image_rgb = image.squeeze() # Convert (H, W, 1) to (H, W)
         else: # Possibly already RGB or other format? Attempt direct conversion
              st.warning(f"Attempting to save image with unexpected shape: {image.shape}")
              image_rgb = image

         pil_image = Image.fromarray(image_rgb)
         buf = io.BytesIO()
         pil_image.save(buf, format=format)
         buf.seek(0)
         return buf.getvalue() # Return bytes
     except Exception as e:
         st.error(f"Error saving image to bytes: {e}")
         return None

def create_zip_archive(image_data_list):
     """Creates a zip archive from a list of (filename, image_bytes) tuples."""
     zip_buffer = io.BytesIO()
     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_f:
         for filename, img_bytes in image_data_list:
             if img_bytes:
                 zip_f.writestr(filename, img_bytes)
     zip_buffer.seek(0)
     return zip_buffer.getvalue()

# --- Main Application ---

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="DIP-Lib: Interactive Image Processing",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🖼️ DIP-Lib: Interactive Image Processing")
    st.markdown("""
    Experiment with common Digital Image Processing techniques!
    Upload an image, select processing modules from the sidebar, adjust parameters,
    add them to the pipeline, and see the results step-by-step.
    """)

    # --- Layout ---
    # Sidebar for controls, Main area for images and results
    sidebar = st.sidebar
    col_main_1, col_main_2 = st.columns([1, 2]) # Adjust ratio as needed

    # --- Sidebar Controls ---
    with sidebar:
        st.header("⚙️ Control Panel")

        # Image upload
        uploaded_file = st.file_uploader("1. Upload Image", type=["jpg", "jpeg", "png"])

        # Load image and handle errors
        if uploaded_file:
            if 'original_image' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
                 # Load only if it's a new file or not loaded yet
                 with st.spinner("Loading image..."):
                    image_bgr, error_msg = load_image(uploaded_file)
                    if image_bgr is not None:
                        st.session_state.original_image = image_bgr
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.session_state.pipeline_steps = [] # Reset pipeline on new image
                        st.success("Image loaded successfully!")
                    else:
                        # Error handled in load_image
                        if 'original_image' in st.session_state: del st.session_state.original_image # Clear invalid image
                        if 'uploaded_filename' in st.session_state: del st.session_state.uploaded_filename
            # else: use cached image in session_state.original_image

        else: # Clear image if uploader is cleared
             if 'original_image' in st.session_state: del st.session_state.original_image
             if 'uploaded_filename' in st.session_state: del st.session_state.uploaded_filename
             if 'pipeline_steps' in st.session_state: st.session_state.pipeline_steps = []

        # --- Module Selection and Pipeline Builder (only if image loaded) ---
        if 'original_image' in st.session_state:
            st.subheader("2. Pipeline Builder")

            # Initialize pipeline steps in session state if not exist
            if 'pipeline_steps' not in st.session_state:
                st.session_state.pipeline_steps = []

            # --- Module Selection ---
            selected_module = st.selectbox("Select Module to Configure", AVAILABLE_MODULES)

            # --- Module Parameters UI ---
            # Build UI dynamically and get current parameters for the selected module
            current_params = build_module_ui(selected_module)

            # --- Add to Pipeline Button ---
            # Basic Validation Example: Check if kernel size is odd if required by a module
            is_valid = True
            validation_error = ""
            # Add more specific validations here if needed
            # Example:
            # if selected_module == "Sharpening" and current_params.get('kernel_size', 1) % 2 == 0:
            #    is_valid = False
            #    validation_error = "Sharpening kernel size must be odd."

            if not is_valid:
                 st.warning(f"Invalid parameters for {selected_module}: {validation_error}")

            if st.button(f"Add '{selected_module}' to Pipeline", key=f"add_{selected_module}", disabled=not is_valid):
                 # Map module name to function
                 func = None
                 if selected_module == "Downsampling & Interpolation": func = ip.downsample_interpolate
                 elif selected_module == "Geometric Transformations": func = ip.geometric_transform
                 elif selected_module == "Add Noise": func = ip.add_noise
                 elif selected_module == "Remove Noise": func = ip.remove_noise
                 elif selected_module == "Image Enhancement": func = ip.enhance_image
                 elif selected_module == "Lighting Correction": func = ip.correct_lighting
                 elif selected_module == "Edge Detection": func = ip.detect_edges
                 elif selected_module == "Sharpening": func = ip.sharpen_image
                 elif selected_module == "Thresholding": func = ip.apply_threshold
                 elif selected_module == "Color Space Conversion":
                      # Check if the function exists before adding
                      if hasattr(ip, 'convert_color_space'):
                           func = ip.convert_color_space
                      else:
                           st.error(f"Cannot add '{selected_module}': Backend function 'ip.convert_color_space' is missing.")
                 # Add mappings for other implemented modules...

                 if func is not None:
                     # Add a copy of the parameters to avoid issues with session state linkage
                     st.session_state.pipeline_steps.append((func, current_params.copy(), selected_module))
                     st.success(f"Added '{selected_module}'!")
                     # Automatically select next common module maybe? or clear params? (Optional UX)
                 elif selected_module not in ["Color Space Conversion"]: # Don't show error if it was expected missing func
                     st.error(f"Error: Could not find function mapping for '{selected_module}'. Check main.py.")

            st.markdown("---") # Separator

            # --- Current Pipeline Display & Management ---
            if st.session_state.pipeline_steps:
                st.subheader("3. Current Pipeline")

                # Display steps with index
                for i, (_, _, module_name) in enumerate(st.session_state.pipeline_steps):
                    st.write(f"{i+1}. {module_name}")

                # --- Remove Step ---
                col_rem1, col_rem2 = st.columns([3,1])
                with col_rem1:
                     # Need unique keys if multiple selectboxes exist
                     step_indices_to_remove = list(range(1, len(st.session_state.pipeline_steps) + 1))
                     selected_step_index = st.selectbox(
                         "Select step to remove",
                         options=step_indices_to_remove,
                         format_func=lambda x: f"Step {x}: {st.session_state.pipeline_steps[x-1][2]}",
                         index=len(step_indices_to_remove) - 1, # Default to last step
                         key="remove_step_select"
                     )
                with col_rem2:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    if st.button("Remove", key="remove_step_button"):
                        removed_module = st.session_state.pipeline_steps.pop(selected_step_index - 1)[2]
                        st.success(f"Removed step {selected_step_index} ('{removed_module}')")
                        st.rerun() # Rerun to update pipeline display

                # --- Clear Pipeline ---
                if st.button("Clear Entire Pipeline", key="clear_pipeline"):
                    st.session_state.pipeline_steps = []
                    st.success("Pipeline cleared!")
                    st.rerun()

                st.markdown("---")
                # --- Execute Pipeline Button ---
                if st.button("🚀 Execute Pipeline", key="execute_pipeline", type="primary"):
                     st.session_state.run_pipeline = True # Use flag to trigger execution in main area
                else:
                     st.session_state.run_pipeline = False

            else:
                st.info("Pipeline is empty. Add modules using the button above.")


    # --- Main Area Display ---
    if 'original_image' in st.session_state:
         original_image = st.session_state.original_image
         # Convert BGR to RGB for displaying with streamlit
         if len(original_image.shape) == 3 and original_image.shape[2] == 3:
             display_image_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
         else:
             display_image_orig = original_image # Grayscale

         # --- Column 1: Original Image and Info ---
         with col_main_1:
             st.subheader("Original Image")
             st.image(display_image_orig, use_container_width=True)
             st.write(f"Shape: {original_image.shape}, Data Type: {original_image.dtype}")
             # Placeholder for ROI selection comment
             st.caption("[Future Idea: ROI selection tool]")

         # --- Column 2: Pipeline Results ---
         with col_main_2:
             # Check if pipeline execution was triggered
             if st.session_state.get('run_pipeline', False) and st.session_state.pipeline_steps:
                 st.subheader("Pipeline Results")

                 # --- Execute Pipeline ---
                 final_image, intermediate_results, step_info_list = apply_pipeline(
                     original_image,
                     st.session_state.pipeline_steps
                 )

                 # --- Display Intermediate Steps with Thumbnails ---
                 st.write("#### Pipeline Steps Visualization")
                 # Calculate number of columns dynamically (e.g., max 4 per row)
                 num_steps_display = len(intermediate_results)
                 cols_per_row = min(4, num_steps_display)
                 num_rows = (num_steps_display + cols_per_row - 1) // cols_per_row

                 img_idx = 0
                 for r in range(num_rows):
                      cols = st.columns(cols_per_row)
                      for c in range(cols_per_row):
                          if img_idx < num_steps_display:
                               step_name, step_image = intermediate_results[img_idx]
                               with cols[c]:
                                   # Convert step image BGR->RGB if needed
                                   if len(step_image.shape) == 3 and step_image.shape[2] == 3:
                                       display_img_step = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
                                   else:
                                       display_img_step = step_image

                                   st.image(display_img_step, caption=step_name, use_container_width=True) # Use column width for auto-scaling
                               img_idx += 1

                 st.markdown("---")

                 # --- Display Final vs Original Side-by-Side ---
                 st.write("#### Final Comparison")
                 col_comp1, col_comp2 = st.columns(2)
                 with col_comp1:
                     st.image(display_image_orig, caption="Original", use_container_width=True)
                 with col_comp2:
                     # Convert final image BGR->RGB if needed
                     if len(final_image.shape) == 3 and final_image.shape[2] == 3:
                         display_img_final = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                     else:
                         display_img_final = final_image
                     st.image(display_img_final, caption="Final Result", use_container_width=True)


                 # --- Metrics ---
                 st.write("#### Quality Metrics (Final vs Original)")
                 # Use the imported calculate_metrics function
                 metrics = calculate_metrics(original_image, final_image)
                 if 'error' in metrics:
                      st.warning(f"Could not calculate metrics: {metrics['error']}")
                 elif 'warning' in metrics:
                      st.warning(f"Metrics Warning: {metrics['warning']}")

                 psnr_val = metrics.get('psnr', 'N/A')
                 ssim_val = metrics.get('ssim', 'N/A')
                 if isinstance(psnr_val, (int, float)): psnr_val = f"{psnr_val:.2f} dB"
                 if isinstance(ssim_val, (int, float)): ssim_val = f"{ssim_val:.4f}"

                 st.metric(label="PSNR", value=psnr_val)
                 st.metric(label="SSIM", value=ssim_val)
                 st.caption("Higher PSNR and SSIM (closer to 1) generally mean better similarity to the original.")


                 # --- Download Options ---
                 st.write("#### Download Results")
                 # Download Final Image
                 final_image_bytes = save_image_to_bytes(final_image, format="PNG")
                 if final_image_bytes:
                     st.download_button(
                         label="Download Final Image (PNG)",
                         data=final_image_bytes,
                         file_name="dip_lib_final_result.png",
                         mime="image/png"
                     )

                 # Download All Steps as ZIP
                 if len(intermediate_results) > 1:
                     image_data_to_zip = []
                     for i, (step_name, step_image) in enumerate(intermediate_results):
                          # Sanitize filename
                          safe_step_name = "".join(c for c in step_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                          filename = f"{i:02d}_{safe_step_name}.png"
                          img_bytes = save_image_to_bytes(step_image, format="PNG")
                          if img_bytes:
                              image_data_to_zip.append((filename, img_bytes))

                     if image_data_to_zip:
                          zip_bytes = create_zip_archive(image_data_to_zip)
                          st.download_button(
                              label="Download All Steps (ZIP)",
                              data=zip_bytes,
                              file_name="dip_lib_pipeline_steps.zip",
                              mime="application/zip"
                          )

                 # Reset flag after processing
                 st.session_state.run_pipeline = False

             # --- Placeholder or instructions if pipeline not run ---
             elif not st.session_state.get('run_pipeline', False) and st.session_state.pipeline_steps:
                 st.info("Pipeline steps are defined. Click 'Execute Pipeline' in the sidebar to see results.")
             else:
                 st.info("Configure and add steps to the pipeline using the sidebar controls.")

    # --- Initial state if no image uploaded ---
    else:
         with col_main_1:
             st.info("⬅️ Upload an image using the sidebar to begin.")
         with col_main_2:
             st.markdown("##### Example Workflow:")
             st.markdown("""
             1.  Upload an image (sidebar).
             2.  Select a module like 'Remove Noise'.
             3.  Adjust parameters (e.g., Filter Type).
             4.  Click 'Add to Pipeline'.
             5.  Select another module like 'Edge Detection'.
             6.  Adjust parameters.
             7.  Click 'Add to Pipeline'.
             8.  Click 'Execute Pipeline' to see the results here.
             """)


if __name__ == "__main__":
    main()