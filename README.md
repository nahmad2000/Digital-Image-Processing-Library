# DIP-Lib: Your Interactive Digital Image Processing Playground!

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digital-image-processing-library-nahmad.streamlit.app/)

**Try it live:** 👉 [https://digital-image-processing-library-nahmad.streamlit.app/](https://digital-image-processing-library-nahmad.streamlit.app/)

## What is DIP-Lib?

DIP-Lib is an easy-to-use web application that brings common image processing tools together in one place. Think of it as a visual laboratory where you can experiment with **eight core digital image processing functionalities** and see the results instantly!

Whether you're learning about image processing, want to quickly test algorithms, or need a tool for a presentation, DIP-Lib helps you:

* **See algorithms in action:** Adjust parameters with sliders and see how images change in real-time.
* **Experiment quickly:** Stack multiple operations (like noise reduction followed by edge detection), reorder them, or remove them to compare results.
* **Understand the process:** Visualize the output of each step in your custom image processing pipeline.

---

## How to Use the Live App

Using DIP-Lib is simple:

1.  **Go to the App:** Click this link -> [https://digital-image-processing-library-nahmad.streamlit.app/](https://digital-image-processing-library-nahmad.streamlit.app/)
2.  **Upload Your Image:** Use the file uploader in the sidebar ("Control Panel") to select an image from your computer (JPG, JPEG, or PNG).
3.  **Select a Module:** In the sidebar under "Pipeline Builder", choose an image processing module you want to try (e.g., "Image Enhancement", "Edge Detection").
4.  **Adjust Parameters:** Use the sliders and options that appear below the module selection to fine-tune the effect.
5.  **Preview (Optional):** Click the "Preview Effect" button to see how the *single selected operation* affects your original image.
6.  **Add to Pipeline:** If you like the effect, click "Add to Pipeline". This adds the operation as a step.
7.  **Build Your Pipeline:** Repeat steps 3-6 to add more processing steps. You can see your pipeline steps listed below the buttons.
8.  **Manage Pipeline:** Remove unwanted steps using the "Select step to remove" dropdown and the "Remove Selected Step" button.
9.  **Execute Pipeline:** Once your pipeline is ready, click "Execute Pipeline". You'll see the original image and the output after each step, plus the final result.
10. **Download:** Click the "Download Processed Image" button to save the final result to your computer.
11. **Clear:** Click "Clear Pipeline" to start over with a fresh pipeline.

## Core Modules Included

You can experiment with these 8 image processing techniques:

* **Downsampling & Interpolation:** See how resizing images affects quality using different algorithms.
* **Geometric Transformations:** Rotate, scale, translate, and shear your images.
* **Noise Analysis & Removal:** Add noise (Gaussian, Salt & Pepper) and try filters (Gaussian Blur, Median, Non-Local Means) to clean it up.
* **Image Enhancement:** Adjust brightness (Gamma) and contrast (Histogram Equalization, CLAHE).
* **Lighting Correction:** Fix uneven lighting using different methods.
* **Edge Detection:** Compare popular edge detection algorithms (Sobel, Scharr, Laplacian, Canny) side-by-side.
* **Sharpening:** Enhance image details and edges using Unsharp Masking.
* **Thresholding:** Convert images to binary (black and white) based on pixel intensity using global or adaptive methods.

## Running Locally (Optional)

If you want to run this application on your own computer:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/nahmad2000/Digital-Image-Processing-Library.git
    cd Digital-Image-Processing-Library
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires: streamlit, opencv-python, numpy, matplotlib, scikit-image, pandas, seaborn)*
4.  **Run the App:**
    ```bash
    streamlit run main.py
    ```
5.  Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

> Enjoy exploring the world of digital image processing!