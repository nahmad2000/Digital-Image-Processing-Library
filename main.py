"""
DIP-Lib: Digital Image Processing Library
Main Streamlit application.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import io

# Import our modules
import image_processors as ip
from utils import plot_comparison, plot_metrics, plot_histogram, convert_to_streamlit

def load_image(uploaded_file):
    """Load an image from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        numpy.ndarray: Loaded image
    """
    if uploaded_file is None:
        return None
    
    # Read the file
    image = Image.open(uploaded_file)
    
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array

def build_module_ui(module_name):
    """Build the UI components for a specific module.
    
    Args:
        module_name (str): Name of the module to build UI for
        
    Returns:
        dict: Parameters selected by the user
    """
    params = {}
    
    if module_name == "Downsampling & Interpolation":
        st.subheader("Downsampling & Interpolation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            params['downsample_method'] = st.selectbox(
                "Downsampling Method",
                options=["simple", "antialias", "area"],
                index=2
            )
            
            params['scale_factor'] = st.slider(
                "Scale Factor",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
        
        with col2:
            params['interpolation_method'] = st.selectbox(
                "Interpolation Method",
                options=["nearest", "bilinear", "bicubic", "lanczos"],
                index=2
            )
    
    elif module_name == "Geometric Transformations":
        st.subheader("Geometric Transformation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            params['rotation'] = st.slider(
                "Rotation (degrees)",
                min_value=-180,
                max_value=180,
                value=0
            )
            
            params['scale_x'] = st.slider(
                "Scale X",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            params['scale_y'] = st.slider(
                "Scale Y",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
        with col2:
            params['translate_x'] = st.slider(
                "Translate X",
                min_value=-100,
                max_value=100,
                value=0
            )
            
            params['translate_y'] = st.slider(
                "Translate Y",
                min_value=-100,
                max_value=100,
                value=0
            )
            
            params['shear_x'] = st.slider(
                "Shear X",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
            
            params['shear_y'] = st.slider(
                "Shear Y",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
    
    elif module_name == "Noise Analysis & Removal":
        st.subheader("Noise Analysis & Removal Settings")
        
        # Noise addition
        st.write("Noise Addition")
        col1, col2 = st.columns(2)
        with col1:
            params['add_noise'] = st.checkbox("Add Noise", value=True)
            params['noise_type'] = st.selectbox(
                "Noise Type",
                options=["gaussian", "salt_pepper"],
                index=0
            )
        
        with col2:
            params['noise_amount'] = st.slider(
                "Noise Amount",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01
            )
        
        # Noise removal
        st.write("Noise Removal")
        col1, col2 = st.columns(2)
        with col1:
            params['remove_noise'] = st.checkbox("Remove Noise", value=True)
            params['filter_type'] = st.selectbox(
                "Filter Type",
                options=["gaussian", "median", "nlm"],
                index=1
            )
        
        with col2:
            if params['filter_type'] in ['gaussian', 'median']:
                params['kernel_size'] = st.slider(
                    "Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2
                )
                
                if params['filter_type'] == 'gaussian':
                    params['sigma'] = st.slider(
                        "Sigma",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.5,
                        step=0.1
                    )
            
            elif params['filter_type'] == 'nlm':
                params['h'] = st.slider(
                    "Filter Strength (h)",
                    min_value=1,
                    max_value=20,
                    value=10,
                    step=1
                )
    
    elif module_name == "Image Enhancement":
        st.subheader("Image Enhancement Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            params['gamma'] = st.slider(
                "Gamma",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1
            )
            
            params['use_equalization'] = st.checkbox("Use Histogram Equalization", value=False)
        
        with col2:
            params['use_clahe'] = st.checkbox("Use CLAHE", value=False)
            
            if params['use_clahe']:
                params['clip_limit'] = st.slider(
                    "CLAHE Clip Limit",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5
                )
                
                params['tile_grid_size'] = st.slider(
                    "CLAHE Tile Size",
                    min_value=2,
                    max_value=16,
                    value=8,
                    step=2
                )
                # Convert to tuple
                params['tile_grid_size'] = (params['tile_grid_size'], params['tile_grid_size'])
    
    elif module_name == "Lighting Correction":
        st.subheader("Lighting Correction Settings")
        
        params['method'] = st.radio(
            "Correction Method",
            options=["spatial", "frequency"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if params['method'] == 'spatial':
                params['kernel_size'] = st.slider(
                    "Kernel Size",
                    min_value=11,
                    max_value=101,
                    value=51,
                    step=10
                )
            
            elif params['method'] == 'frequency':
                params['gamma'] = st.slider(
                    "Gamma",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1
                )
        
        with col2:
            if params['method'] == 'frequency':
                params['cutoff_low'] = st.slider(
                    "Low Frequency Gain",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
                
                params['cutoff_high'] = st.slider(
                    "High Frequency Gain",
                    min_value=1.1,
                    max_value=3.0,
                    value=2.0,
                    step=0.1
                )
    
    elif module_name == "Edge Detection":
        st.subheader("Edge Detection Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            params['method'] = st.selectbox(
                "Edge Detection Method",
                options=["sobel", "scharr", "laplacian", "canny"],
                index=3
            )
            
            if params['method'] != 'canny':
                params['threshold1'] = st.slider(
                    "Threshold",
                    min_value=10,
                    max_value=250,
                    value=100,
                    step=10
                )
            
            else:
                params['threshold1'] = st.slider(
                    "Lower Threshold",
                    min_value=10,
                    max_value=250,
                    value=100,
                    step=10
                )
                
                params['threshold2'] = st.slider(
                    "Upper Threshold",
                    min_value=10,
                    max_value=250,
                    value=200,
                    step=10
                )
        
        with col2:
            params['aperture_size'] = st.select_slider(
                "Aperture Size",
                options=[3, 5, 7],
                value=3
            )
            
            params['L2gradient'] = st.checkbox("Use L2 Gradient", value=False)
    
    return params

def apply_pipeline(image, pipeline_steps):
    """Apply a sequence of processing steps to an image.
    
    Args:
        image (numpy.ndarray): Input image
        pipeline_steps (list): List of (function, params, module_name) tuples
        
    Returns:
        numpy.ndarray: Processed image
        list: List of intermediate results
    """
    current_image = image.copy()
    intermediate_results = [("Original", image)]
    
    for i, (func, params, module_name) in enumerate(pipeline_steps):
        # Apply the function with parameters
        if module_name == "Downsampling & Interpolation":
            result = func(current_image, **params)
            current_image = result['image']
            intermediate_results.append((f"Step {i+1}: {module_name}", current_image))
        
        elif module_name == "Noise Analysis & Removal":
            # First add noise if requested
            if params.get('add_noise', False):
                noisy_image = ip.add_noise(
                    current_image, 
                    noise_type=params['noise_type'], 
                    amount=params['noise_amount']
                )
                intermediate_results.append((f"Step {i+1}a: Add {params['noise_type']} Noise", noisy_image))
                
                # Then apply filter if requested
                if params.get('remove_noise', False):
                    filter_params = {}
                    if 'kernel_size' in params:
                        filter_params['kernel_size'] = params['kernel_size']
                    if 'sigma' in params:
                        filter_params['sigma'] = params['sigma']
                    if 'h' in params:
                        filter_params['h'] = params['h']
                    
                    filtered_image = ip.remove_noise(
                        noisy_image, 
                        filter_type=params['filter_type'], 
                        params=filter_params
                    )
                    current_image = filtered_image
                    intermediate_results.append((f"Step {i+1}b: {params['filter_type']} Filter", current_image))
                else:
                    current_image = noisy_image
            else:
                # Just apply filter to original image
                if params.get('remove_noise', False):
                    filter_params = {}
                    if 'kernel_size' in params:
                        filter_params['kernel_size'] = params['kernel_size']
                    if 'sigma' in params:
                        filter_params['sigma'] = params['sigma']
                    if 'h' in params:
                        filter_params['h'] = params['h']
                    
                    filtered_image = ip.remove_noise(
                        current_image, 
                        filter_type=params['filter_type'], 
                        params=filter_params
                    )
                    current_image = filtered_image
                    intermediate_results.append((f"Step {i+1}: {params['filter_type']} Filter", current_image))
        
        else:
            # For all other modules, just apply the function
            result = func(current_image, **params)
            current_image = result
            intermediate_results.append((f"Step {i+1}: {module_name}", current_image))
    
    return current_image, intermediate_results

def save_results(image, filename="processed_image.jpg"):
    """Save processed image.
    
    Args:
        image (numpy.ndarray): Processed image
        filename (str): Filename to save as
    
    Returns:
        bytes: Image bytes for download
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)
    
    return buf

def main():
    """Main application entry point."""
    # Set page title and layout
    st.set_page_config(
        page_title="DIP-Lib: Digital Image Processing Library",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("DIP-Lib: Digital Image Processing Library")
    
    # Add description
    st.markdown("""
    DIP-Lib unifies six core digital image processing functionalities into a single, user-friendly pipeline.
    Upload an image, select processing modules, and build your custom pipeline.
    """)
    
    # Sidebar for image upload and module selection
    with st.sidebar:
        st.header("Control Panel")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        # Module selection
        st.subheader("Pipeline Builder")
        
        # Initialize pipeline steps in session state if not exist
        if 'pipeline_steps' not in st.session_state:
            st.session_state.pipeline_steps = []
        
        # Available modules
        available_modules = [
            "Downsampling & Interpolation",
            "Geometric Transformations",
            "Noise Analysis & Removal",
            "Image Enhancement",
            "Lighting Correction",
            "Edge Detection"
        ]
        
        selected_module = st.selectbox("Select Module", available_modules)
    
    # Load image
    image = None
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        if image is not None:
            # Display original image
            st.subheader("Original Image")
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image
            st.image(display_image, width=400)
            
            # Show image info
            st.write(f"Image Shape: {image.shape}")
            
            # Display module UI based on selection
            params = build_module_ui(selected_module)
            
            # Preview button
            if st.button("Preview Effect"):
                st.subheader(f"Preview: {selected_module}")
                
                # Apply selected module
                if selected_module == "Downsampling & Interpolation":
                    result = ip.downsample_interpolate(image, **params)
                    processed_image = result['image']
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                caption="Original", width=400)
                    with col2:
                        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB) if len(processed_image.shape) == 3 and processed_image.shape[2] == 3 else processed_image, 
                                caption="Processed", width=400)
                    
                    # Display metrics
                    st.write("Image Quality Metrics:")
                    st.write(f"PSNR: {result['metrics'].get('psnr', 'N/A'):.2f} dB")
                    st.write(f"SSIM: {result['metrics'].get('ssim', 'N/A'):.4f}")
                
                elif selected_module == "Geometric Transformations":
                    processed_image = ip.geometric_transform(image, **params)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                caption="Original", width=400)
                    with col2:
                        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB) if len(processed_image.shape) == 3 and processed_image.shape[2] == 3 else processed_image, 
                                caption="Transformed", width=400)
                
                elif selected_module == "Noise Analysis & Removal":
                    # First add noise if requested
                    if params.get('add_noise', False):
                        noisy_image = ip.add_noise(
                            image, 
                            noise_type=params['noise_type'], 
                            amount=params['noise_amount']
                        )
                        
                        # Display noise comparison
                        st.write("Noise Addition:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                    caption="Original", width=400)
                        with col2:
                            st.image(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB) if len(noisy_image.shape) == 3 and noisy_image.shape[2] == 3 else noisy_image, 
                                    caption=f"{params['noise_type']} Noise", width=400)
                        
                        # Calculate noise metrics
                        metrics = ip.calculate_metrics(image, noisy_image)
                        st.write(f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
                        st.write(f"SSIM: {metrics.get('ssim', 'N/A'):.4f}")
                        
                        # Then apply filter if requested
                        if params.get('remove_noise', False):
                            filter_params = {}
                            if 'kernel_size' in params:
                                filter_params['kernel_size'] = params['kernel_size']
                            if 'sigma' in params:
                                filter_params['sigma'] = params['sigma']
                            if 'h' in params:
                                filter_params['h'] = params['h']
                            
                            filtered_image = ip.remove_noise(
                                noisy_image, 
                                filter_type=params['filter_type'], 
                                params=filter_params
                            )
                            
                            # Display filter comparison
                            st.write("Noise Removal:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB) if len(noisy_image.shape) == 3 and noisy_image.shape[2] == 3 else noisy_image, 
                                        caption="Noisy Image", width=400)
                            with col2:
                                st.image(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB) if len(filtered_image.shape) == 3 and filtered_image.shape[2] == 3 else filtered_image, 
                                        caption=f"{params['filter_type']} Filtered", width=400)
                            
                            # Calculate filter metrics
                            metrics = ip.calculate_metrics(image, filtered_image)
                            st.write(f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
                            st.write(f"SSIM: {metrics.get('ssim', 'N/A'):.4f}")
                            
                            processed_image = filtered_image
                        else:
                            processed_image = noisy_image
                    else:
                        # Just apply filter to original image
                        if params.get('remove_noise', False):
                            filter_params = {}
                            if 'kernel_size' in params:
                                filter_params['kernel_size'] = params['kernel_size']
                            if 'sigma' in params:
                                filter_params['sigma'] = params['sigma']
                            if 'h' in params:
                                filter_params['h'] = params['h']
                            
                            filtered_image = ip.remove_noise(
                                image, 
                                filter_type=params['filter_type'], 
                                params=filter_params
                            )
                            
                            # Display filter comparison
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                        caption="Original", width=400)
                            with col2:
                                st.image(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB) if len(filtered_image.shape) == 3 and filtered_image.shape[2] == 3 else filtered_image, 
                                        caption=f"{params['filter_type']} Filtered", width=400)
                            
                            processed_image = filtered_image
                        else:
                            processed_image = image
                
                elif selected_module == "Image Enhancement":
                    processed_image = ip.enhance_image(image, **params)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                caption="Original", width=400)
                    with col2:
                        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB) if len(processed_image.shape) == 3 and processed_image.shape[2] == 3 else processed_image, 
                                caption="Enhanced", width=400)
                    
                    # Display histograms
                    st.write("Histograms:")
                    col1, col2 = st.columns(2)
                    with col1:
                        orig_hist_fig = plot_histogram(image)
                        st.pyplot(orig_hist_fig)
                    with col2:
                        proc_hist_fig = plot_histogram(processed_image)
                        st.pyplot(proc_hist_fig)
                
                elif selected_module == "Lighting Correction":
                    processed_image = ip.correct_lighting(image, **params)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                caption="Original", width=400)
                    with col2:
                        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB) if len(processed_image.shape) == 3 and processed_image.shape[2] == 3 else processed_image, 
                                caption="Lighting Corrected", width=400)
                
                elif selected_module == "Edge Detection":
                    processed_image = ip.detect_edges(image, **params)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image, 
                                caption="Original", width=400)
                    with col2:
                        st.image(processed_image, caption=f"{params['method']} Edges", width=400)
            
            # Add to pipeline button
            if st.button("Add to Pipeline"):
                # Create function and params tuple based on selected module
                if selected_module == "Downsampling & Interpolation":
                    func = ip.downsample_interpolate
                elif selected_module == "Geometric Transformations":
                    func = ip.geometric_transform
                elif selected_module == "Noise Analysis & Removal":
                    # Special case, will be handled in apply_pipeline
                    func = None
                elif selected_module == "Image Enhancement":
                    func = ip.enhance_image
                elif selected_module == "Lighting Correction":
                    func = ip.correct_lighting
                elif selected_module == "Edge Detection":
                    func = ip.detect_edges
                
                # Add to pipeline steps
                st.session_state.pipeline_steps.append((func, params, selected_module))
                st.success(f"Added {selected_module} to pipeline!")
            
            # Display current pipeline
            if st.session_state.pipeline_steps:
                st.subheader("Current Pipeline")
                
                # Display steps
                for i, (_, _, module_name) in enumerate(st.session_state.pipeline_steps):
                    st.write(f"{i+1}. {module_name}")
                
                # Option to remove a step
                step_to_remove = st.selectbox(
                    "Select step to remove",
                    options=list(range(1, len(st.session_state.pipeline_steps) + 1)),
                    format_func=lambda x: f"Step {x}: {st.session_state.pipeline_steps[x-1][2]}"
                )
                
                if st.button("Remove Selected Step"):
                    # Remove the step (index is 0-based but display is 1-based)
                    st.session_state.pipeline_steps.pop(step_to_remove - 1)
                    st.success(f"Removed step {step_to_remove}")
                    st.rerun()
                
                # Execute pipeline
                if st.button("Execute Pipeline"):
                    st.subheader("Pipeline Results")
                    
                    # Apply pipeline
                    final_image, intermediate_results = apply_pipeline(image, st.session_state.pipeline_steps)
                    
                    # Display results
                    st.write("#### Pipeline Steps")
                    cols = st.columns(min(3, len(intermediate_results)))
                    
                    for i, (step_name, step_image) in enumerate(intermediate_results):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            if len(step_image.shape) == 3 and step_image.shape[2] == 3:
                                display_img = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
                            else:
                                display_img = step_image
                            st.image(display_img, caption=step_name, width=400)
                    
                    # Calculate metrics for final result
                    if len(intermediate_results) > 1:
                        metrics = ip.calculate_metrics(image, final_image)
                        st.write("#### Final Metrics")
                        st.write(f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
                        st.write(f"SSIM: {metrics.get('ssim', 'N/A'):.4f}")
                    
                    # Download button
                    st.write("#### Download Result")
                    result_bytes = save_results(final_image)
                    st.download_button(
                        label="Download Processed Image",
                        data=result_bytes,
                        file_name="dip_lib_result.jpg",
                        mime="image/jpeg"
                    )
                
                # Clear pipeline button
                if st.button("Clear Pipeline"):
                    st.session_state.pipeline_steps = []
                    st.success("Pipeline cleared!")
                    st.rerun()
        
        else:
            st.error("Failed to load image. Please try another file.")
    
    else:
        # Show a placeholder or demo if no image is uploaded
        st.info("Please upload an image to start.")

if __name__ == "__main__":
    main()