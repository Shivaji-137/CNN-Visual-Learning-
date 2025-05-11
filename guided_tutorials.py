import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import io
import time
import cv2
import pandas as pd
from scipy.signal import convolve2d

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# Function to create a simple sample image for convolution demonstration
def create_sample_image(type="edge", size=32):
    """Create a sample image for convolution demonstration"""
    img = np.zeros((size, size))
    
    if type == "edge":
        # Create a simple edge
        img[size//4:3*size//4, size//2:] = 1.0
    elif type == "corner":
        # Create a simple corner
        img[size//4:3*size//4, size//4:3*size//4] = 1.0
    elif type == "gradient":
        # Create a gradient
        for i in range(size):
            img[i, :] = i / size
    elif type == "circle":
        # Create a circle
        center = size // 2
        radius = size // 4
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 < radius ** 2:
                    img[i, j] = 1.0
    elif type == "cross":
        # Create a cross
        img[size//4:3*size//4, size//2-1:size//2+2] = 1.0
        img[size//2-1:size//2+2, size//4:3*size//4] = 1.0
    
    return img

# Common CNN filters
COMMON_FILTERS = {
    "Horizontal Edge Detection": np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ]),
    "Vertical Edge Detection": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    "Sobel Horizontal": np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]),
    "Sobel Vertical": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    "Gaussian Blur": np.array([
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
    ]),
    "Sharpen": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    "Ridge Detection": np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
}

def apply_convolution(image, kernel):
    """Apply convolution with the given kernel"""
    # Ensure the image is a 2D array
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)  # Convert to grayscale
    
    # Apply convolution
    result = convolve2d(image, kernel, mode='same', boundary='symm')
    
    # Normalize to [0, 1] range for better visualization
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    
    return result

def apply_pooling(image, pool_type="max", pool_size=2, stride=2):
    """Apply pooling operation"""
    # Get image dimensions
    h, w = image.shape
    
    # Output dimensions
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    
    # Initialize output
    result = np.zeros((h_out, w_out))
    
    # Apply pooling
    for i in range(h_out):
        for j in range(w_out):
            # Extract region
            i_start = i * stride
            j_start = j * stride
            region = image[i_start:i_start+pool_size, j_start:j_start+pool_size]
            
            if pool_type == "max":
                result[i, j] = np.max(region)
            elif pool_type == "avg":
                result[i, j] = np.mean(region)
    
    return result

def apply_activation(feature_map, activation_type="relu"):
    """Apply activation function"""
    if activation_type == "relu":
        return np.maximum(0, feature_map)
    elif activation_type == "sigmoid":
        return 1 / (1 + np.exp(-feature_map))
    elif activation_type == "tanh":
        return np.tanh(feature_map)
    else:
        return feature_map

def guided_tutorials():
    st.title("Guided CNN Tutorials")
    
    st.markdown("""
    # Interactive CNN Tutorials
    
    Explore these interactive tutorials to understand how Convolutional Neural Networks work.
    Each tutorial focuses on a specific aspect of CNNs with interactive visualizations and step-by-step explanations.

    ## Tutorial Topics:
    - Convolution Operation: Learn how filters detect features in images
    - Activation Functions: Understand the role of non-linearity in neural networks
    - Pooling Operations: See how spatial dimensions are reduced while preserving features
    - CNN Architecture Flow: Follow data through a complete CNN pipeline
    """)
    
    # Tutorial selection
    tutorial = st.selectbox(
        "Select a Tutorial:",
        ["Convolution Operation", "Activation Functions", "Pooling Operations", "CNN Architecture Flow", "CNN Training Concepts", "Quiz: Test Your Knowledge"]
    )
    
    if tutorial == "Convolution Operation":
        convolution_tutorial()
    elif tutorial == "Activation Functions":
        activation_tutorial()
    elif tutorial == "Pooling Operations":
        pooling_tutorial()
    elif tutorial == "CNN Architecture Flow":
        architecture_flow_tutorial()
    elif tutorial == "CNN Training Concepts":
        training_concepts_tutorial()
    elif tutorial == "Quiz: Test Your Knowledge":
        cnn_quiz()

def convolution_tutorial():
    st.subheader("Understanding Convolution")
    
    st.markdown("""
    ## The Convolution Operation
    
    Convolution is the fundamental operation in CNN that helps detect features like edges, textures, and patterns in images.
    
    ### How it Works:
    1. A small **filter** (or kernel) slides across the input image
    2. At each position, element-wise multiplication and summation is performed
    3. The result is stored in a new output image called a **feature map**
    
    Different filters detect different features. Let's explore some common filters:
    """)
    
    # Interactive convolution demonstration
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample image selection
        image_type = st.selectbox(
            "Select Sample Image:",
            ["edge", "corner", "gradient", "circle", "cross"]
        )
        
        # Create the sample image
        sample_img = create_sample_image(image_type)
        
        # Display the sample image
        st.image(sample_img, caption="Sample Image", use_container_width=True, clamp=True)
    
    with col2:
        # Filter selection
        filter_name = st.selectbox(
            "Select Filter:",
            list(COMMON_FILTERS.keys())
        )
        
        # Get the selected filter
        selected_filter = COMMON_FILTERS[filter_name]
        
        # Display the filter
        st.markdown(f"**{filter_name} Filter:**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(selected_filter, cmap='viridis')
        plt.colorbar(im, ax=ax)
        
        # Add filter values as text
        for i in range(selected_filter.shape[0]):
            for j in range(selected_filter.shape[1]):
                text = ax.text(j, i, f"{selected_filter[i, j]:.2f}",
                       ha="center", va="center", color="w" if abs(selected_filter[i, j]) > 0.5 else "k")
        
        ax.set_title(f"{filter_name}")
        st.pyplot(fig)
    
    # Apply convolution and show result
    st.subheader("Convolution Result")
    
    # Apply the convolution
    conv_result = apply_convolution(sample_img, selected_filter)
    
    # Display the result
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("**Input Image**")
        st.image(sample_img, caption="Input", use_container_width=True, clamp=True)
    
    with col2:
        st.markdown("**Filter**")
        filter_img = (selected_filter - selected_filter.min()) / (selected_filter.max() - selected_filter.min())
        st.image(filter_img, caption=filter_name, use_container_width=True, clamp=True)
    
    with col3:
        st.markdown("**Feature Map**")
        st.image(conv_result, caption="Output", use_container_width=True, clamp=True)
    
    # Visualize convolution in 3D
    st.subheader("3D Visualization of Convolution")
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121, projection='3d')
    
    # Create meshgrid for 3D plot
    x = np.arange(0, sample_img.shape[1])
    y = np.arange(0, sample_img.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Plot the input image in 3D
    surf = ax.plot_surface(X, Y, sample_img, cmap='viridis', alpha=0.8)
    ax.set_title("Input Image")
    
    # Plot the convolution result in 3D
    ax = fig.add_subplot(122, projection='3d')
    surf = ax.plot_surface(X, Y, conv_result, cmap='plasma', alpha=0.8)
    ax.set_title("Convolution Result")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interactive explanation of filter purpose
    st.subheader("Understanding This Filter")
    
    filter_explanations = {
        "Horizontal Edge Detection": """
        This filter detects **horizontal edges** by taking the difference between the pixel values above and below.
        - Positive values in bottom row: Highlights transitions from dark to light (top to bottom)
        - Negative values in top row: Highlights transitions from light to dark (top to bottom)
        - Zero values in middle row: Ignores horizontal lines
        
        **Applications**: Finding horizontal lines in images, text line detection
        """,
        
        "Vertical Edge Detection": """
        This filter detects **vertical edges** by taking the difference between pixel values on the left and right.
        - Positive values in right column: Highlights transitions from dark to light (left to right)
        - Negative values in left column: Highlights transitions from light to dark (left to right)
        - Zero values in middle column: Ignores vertical lines
        
        **Applications**: Finding vertical boundaries, detecting columns or buildings
        """,
        
        "Sobel Horizontal": """
        The **Sobel horizontal filter** is an improved edge detector that adds emphasis to central pixels.
        - Stronger weights in the middle columns (×2): Gives more importance to central pixels
        - Similar to Horizontal Edge Detection but with better noise handling
        
        **Applications**: Image processing pipelines, detecting more defined horizontal edges
        """,
        
        "Sobel Vertical": """
        The **Sobel vertical filter** is an improved edge detector for vertical edges with emphasis on central pixels.
        - Stronger weights in the middle rows (×2): Gives more importance to central pixels
        - Similar to Vertical Edge Detection but with better noise handling
        
        **Applications**: Image processing pipelines, detecting more defined vertical edges
        """,
        
        "Gaussian Blur": """
        The **Gaussian blur filter** averages pixels with their neighbors, giving more weight to closer pixels.
        - Highest weight in center (1/4): Most influence from the center pixel
        - Decreasing weights with distance: Less influence from farther pixels
        - All positive values: Performs weighted averaging
        
        **Applications**: Noise reduction, pre-processing for feature detection, creating smooth effects
        """,
        
        "Sharpen": """
        The **Sharpen filter** enhances details by increasing the weight of the center pixel relative to its neighbors.
        - High positive value in center (5): Emphasizes the center pixel
        - Negative values around: Subtracts neighboring pixels
        - Net effect: Amplifies differences between a pixel and its surroundings
        
        **Applications**: Detail enhancement, making blurry images clearer, emphasizing textures
        """,
        
        "Identity": """
        The **Identity filter** simply passes through the original image values.
        - 1 in center, 0 elsewhere: Only keeps the center pixel value
        - Result is identical to input (hence the name)
        
        **Applications**: Used as a baseline or for certain skip connections in deep networks
        """,
        
        "Ridge Detection": """
        The **Ridge Detection filter** (also called Laplacian) detects areas where intensity changes direction.
        - High positive weight in center (8): Emphasizes the center
        - Negative values around (-1): Subtracts all neighbors
        - Net effect: Highlights regions where the image changes concavity
        
        **Applications**: Finding ridges, detecting complete boundaries, highlighting areas where the gradient changes direction
        """
    }
    
    if filter_name in filter_explanations:
        st.markdown(filter_explanations[filter_name])
    
    # Practical exercise
    st.subheader("Hands-on Exercise")
    
    st.markdown("""
    ### Custom Filter Creation
    
    Now it's your turn! Create a custom filter and see how it affects the image.
    Experiment with different values to understand how each position in the filter influences the result.
    """)
    
    # Create a grid of input fields for a 3x3 filter
    custom_filter = np.zeros((3, 3))
    cols = st.columns(3)
    
    for i in range(3):
        for j in range(3):
            with cols[j]:
                custom_filter[i, j] = st.number_input(
                    f"Filter[{i},{j}]",
                    min_value=-10.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key=f"custom_filter_{i}_{j}"
                )
    
    # Normalize the filter (optional)
    normalize_filter = st.checkbox("Normalize Filter (sum to 1)", value=False)
    if normalize_filter and np.sum(np.abs(custom_filter)) > 0:
        custom_filter = custom_filter / np.sum(np.abs(custom_filter))
    
    # Apply the custom filter
    custom_result = apply_convolution(sample_img, custom_filter)
    
    # Display the result
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Custom Filter**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(custom_filter, cmap='viridis')
        plt.colorbar(im, ax=ax)
        
        # Add filter values as text
        for i in range(custom_filter.shape[0]):
            for j in range(custom_filter.shape[1]):
                text = ax.text(j, i, f"{custom_filter[i, j]:.2f}",
                       ha="center", va="center", color="w" if abs(custom_filter[i, j]) > 0.5 else "k")
        
        ax.set_title("Custom Filter")
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Result with Custom Filter**")
        st.image(custom_result, caption="Custom Filter Result", use_container_width=True, clamp=True)
    
    # Learning checkpoints
    st.subheader("Key Takeaways")
    
    st.markdown("""
    ### What You've Learned:
    
    - **Convolution** involves sliding a filter over an image and computing the sum of element-wise multiplications
    - Different **filters** detect different features (edges, textures, patterns)
    - The resulting **feature maps** highlight specific attributes of the image
    - In CNNs, these filters are **learned automatically** during training
    - Multiple filters create multiple feature maps, allowing the network to detect various features
    
    ### Next Steps:
    
    After convolution, CNNs typically apply an activation function, which introduces non-linearity.
    Proceed to the Activation Functions tutorial to learn more.
    """)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("⬅️ [Return to Tutorial Selection](#guided-cnn-tutorials)")
    with col2:
        st.markdown("➡️ [Next: Activation Functions](#)")

def activation_tutorial():
    st.subheader("Understanding Activation Functions")
    
    st.markdown("""
    ## Activation Functions in CNNs
    
    Activation functions introduce **non-linearity** into neural networks, which is crucial for learning complex patterns.
    Without them, a neural network would just be a series of linear transformations.
    
    ### Why Non-Linearity Matters:
    
    - Enables the network to learn **complex, non-linear relationships** in data
    - Allows **hierarchical feature learning** across multiple layers
    - Makes the network capable of approximating any function (universal approximation theorem)
    
    Let's explore common activation functions and see how they transform feature maps:
    """)
    
    # Interactive activation function demonstration
    col1, col2 = st.columns(2)
    
    with col1:
        # Input selection
        input_type = st.selectbox(
            "Select Input (After Convolution):",
            ["edge", "corner", "gradient", "circle", "cross"]
        )
        
        # Select filter to apply
        filter_name = st.selectbox(
            "Apply Filter:",
            list(COMMON_FILTERS.keys())
        )
        
        # Create the sample image
        sample_img = create_sample_image(input_type)
        
        # Apply convolution
        selected_filter = COMMON_FILTERS[filter_name]
        conv_result = apply_convolution(sample_img, selected_filter)
        
        # Display the input and convolution result
        st.markdown("**Original Image:**")
        st.image(sample_img, caption="Input Image", use_container_width=True, clamp=True)
        
        st.markdown("**After Convolution:**")
        st.image(conv_result, caption=f"Convolved with {filter_name}", use_container_width=True, clamp=True)
    
    with col2:
        # Activation function selection
        activation_type = st.selectbox(
            "Select Activation Function:",
            ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "None (Linear)"]
        )
        
        # Apply activation function
        if activation_type == "ReLU":
            result = apply_activation(conv_result, "relu")
            formula = "ReLU(x) = max(0, x)"
            description = """
            **ReLU (Rectified Linear Unit)**:
            - Outputs the input directly if positive, otherwise outputs zero
            - Simple, computationally efficient
            - Helps with the vanishing gradient problem
            - Most commonly used in modern CNNs
            - Can suffer from "dying ReLU" problem (neurons permanently inactive)
            """
        elif activation_type == "Sigmoid":
            result = apply_activation(conv_result, "sigmoid")
            formula = "Sigmoid(x) = 1 / (1 + e^(-x))"
            description = """
            **Sigmoid**:
            - Squashes values between 0 and 1
            - Smooth, differentiable function
            - Historically popular, now less common in hidden layers
            - Can cause vanishing gradient problem in deep networks
            - Often used for binary classification output layer
            """
        elif activation_type == "Tanh":
            result = apply_activation(conv_result, "tanh")
            formula = "Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"
            description = """
            **Tanh (Hyperbolic Tangent)**:
            - Squashes values between -1 and 1
            - Zero-centered, which helps with optimization
            - Generally performs better than sigmoid for hidden layers
            - Still can suffer from vanishing gradient problem
            - Good for data with both positive and negative values
            """
        elif activation_type == "Leaky ReLU":
            # Custom implementation for Leaky ReLU
            alpha = st.slider("Alpha value (slope for negative inputs):", 0.01, 0.5, 0.1)
            result = np.maximum(alpha * conv_result, conv_result)
            formula = f"LeakyReLU(x) = max(αx, x), where α = {alpha}"
            description = f"""
            **Leaky ReLU**:
            - Similar to ReLU, but allows a small gradient for negative values (α = {alpha})
            - Addresses the "dying ReLU" problem
            - Maintains the computational efficiency of ReLU
            - Better gradient flow during backpropagation
            - Used in many modern architectures as an alternative to ReLU
            """
        else:  # No activation
            result = conv_result
            formula = "f(x) = x"
            description = """
            **Linear (No Activation)**:
            - Simply passes the input value unchanged
            - Provides no non-linearity
            - Multiple stacked linear layers would be equivalent to a single linear layer
            - Rarely used in hidden layers, sometimes in network output for regression
            """
        
        # Display the formula
        st.markdown(f"**Formula:** {formula}")
        
        # Display the result after activation
        st.markdown("**After Activation:**")
        st.image(result, caption=f"After {activation_type}", use_container_width=True, clamp=True)
        
        # Display description
        st.markdown(description)
    
    # 3D visualization of activation effect
    st.subheader("3D Visualization of Activation Effect")
    
    # Setup for the 3D visualization
    x = np.arange(0, conv_result.shape[1])
    y = np.arange(0, conv_result.shape[0])
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # Create the 3D visualization with Plotly
    fig = go.Figure()
    
    # Add the surface for convolution result (before activation)
    fig.add_trace(go.Surface(
        z=conv_result,
        x=x_mesh, 
        y=y_mesh,
        colorscale='Viridis',
        opacity=0.7,
        showscale=False,
        name="Before Activation"
    ))
    
    # Add the surface for activation result
    fig.add_trace(go.Surface(
        z=result,
        x=x_mesh, 
        y=y_mesh,
        colorscale='Plasma',
        opacity=0.9,
        showscale=True,
        colorbar=dict(title="After Activation"),
        name="After Activation"
    ))
    
    fig.update_layout(
        title=f"Effect of {activation_type} Activation",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Activation Value',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig)
    
    # Interactive 1D visualization of activation functions
    st.subheader("Interactive Visualization of Activation Functions")
    
    st.markdown("""
    This graph shows how each activation function transforms input values. 
    Move the slider to see how different input ranges are affected.
    """)
    
    # Range for input values
    x_min = st.slider("Minimum Input Value:", -10.0, 0.0, -5.0)
    x_max = st.slider("Maximum Input Value:", 0.0, 10.0, 5.0)
    
    # Generate input values
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate activation function outputs
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    leaky_relu = np.maximum(0.1 * x, x)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, x, 'k--', label='Linear (No Activation)')
    ax.plot(x, relu, label='ReLU')
    ax.plot(x, sigmoid, label='Sigmoid')
    ax.plot(x, tanh, label='Tanh')
    ax.plot(x, leaky_relu, label='Leaky ReLU (α=0.1)')
    
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('Output (y)')
    ax.set_title('Comparing Activation Functions')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    st.pyplot(fig)
    
    # Practical application of activation functions
    st.subheader("Practical Application: Feature Enhancement")
    
    st.markdown("""
    Activation functions can significantly impact what features are highlighted in an image. 
    Here we see how different activations affect feature extraction:
    """)
    
    # Create a more complex sample image
    complex_img = create_sample_image("circle", size=64)
    
    # Apply a few different filters and activations
    filters_to_show = ["Horizontal Edge Detection", "Vertical Edge Detection", "Ridge Detection"]
    activations_to_show = ["None (Linear)", "ReLU", "Sigmoid"]
    
    # Create a grid to show the results
    fig, axs = plt.subplots(len(filters_to_show), len(activations_to_show) + 1, figsize=(12, 8))
    
    # Add titles to columns
    axs[0, 0].set_title("Filter")
    for j, act in enumerate(activations_to_show):
        axs[0, j+1].set_title(act)
    
    for i, filter_name in enumerate(filters_to_show):
        # Get the filter
        curr_filter = COMMON_FILTERS[filter_name]
        
        # Apply convolution
        conv_result = apply_convolution(complex_img, curr_filter)
        
        # Show the filter
        axs[i, 0].imshow(curr_filter, cmap='viridis')
        axs[i, 0].set_ylabel(filter_name)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        
        # Show results with different activations
        for j, act in enumerate(activations_to_show):
            if act == "None (Linear)":
                result = conv_result
            elif act == "ReLU":
                result = apply_activation(conv_result, "relu")
            elif act == "Sigmoid":
                result = apply_activation(conv_result, "sigmoid")
            
            axs[i, j+1].imshow(result, cmap='plasma')
            axs[i, j+1].set_xticks([])
            axs[i, j+1].set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Learning checkpoints
    st.subheader("Key Takeaways")
    
    st.markdown("""
    ### What You've Learned:
    
    - **Activation functions** add non-linearity to neural networks
    - **ReLU** is most commonly used due to its simplicity and effectiveness
    - Different activations have different properties:
      - **ReLU & Leaky ReLU**: Good for deep networks, help with vanishing gradients
      - **Sigmoid & Tanh**: Squash values to bounded ranges, useful for certain tasks
    - Activations determine which features are emphasized or suppressed
    - Modern CNNs typically use ReLU or variants like Leaky ReLU
    
    ### Next Steps:
    
    After activation, CNNs typically use pooling to reduce spatial dimensions.
    Proceed to the Pooling Operations tutorial to learn more.
    """)
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("⬅️ [Return to Tutorial Selection](#guided-cnn-tutorials)")
    with col2:
        st.markdown("⬅️ [Previous: Convolution Operation](#)")
    with col3:
        st.markdown("➡️ [Next: Pooling Operations](#)")

def pooling_tutorial():
    st.subheader("Understanding Pooling Operations")
    
    st.markdown("""
    ## Pooling in CNNs
    
    Pooling operations reduce the spatial dimensions (width and height) of feature maps while preserving important information.
    
    ### Key Benefits of Pooling:
    
    1. **Dimensionality Reduction** - Reduces computation by decreasing feature map size
    2. **Translation Invariance** - Makes detection less sensitive to exact feature location
    3. **Feature Abstraction** - Helps capture hierarchical patterns at different scales
    4. **Overfitting Prevention** - Reduces model parameters and adds robustness
    
    Let's explore different pooling operations:
    """)
    
    # Interactive pooling demonstration
    col1, col2 = st.columns(2)
    
    with col1:
        # Input selection and processing
        input_type = st.selectbox(
            "Select Input Image:",
            ["circle", "edge", "corner", "gradient", "cross"]
        )
        
        # Create the sample image (making it larger to better show pooling)
        sample_img = create_sample_image(input_type, size=64)
        
        # Apply some convolution to create a more interesting feature map
        filter_name = st.selectbox(
            "Apply Filter Before Pooling:",
            list(COMMON_FILTERS.keys())
        )
        
        # Apply convolution and activation
        selected_filter = COMMON_FILTERS[filter_name]
        conv_result = apply_convolution(sample_img, selected_filter)
        
        # Apply activation function
        activation_type = st.selectbox(
            "Apply Activation Before Pooling:",
            ["None (Linear)", "ReLU", "Sigmoid", "Tanh"]
        )
        
        if activation_type == "ReLU":
            feature_map = apply_activation(conv_result, "relu")
        elif activation_type == "Sigmoid":
            feature_map = apply_activation(conv_result, "sigmoid")
        elif activation_type == "Tanh":
            feature_map = apply_activation(conv_result, "tanh")
        else:
            feature_map = conv_result
        
        # Display the input and processed feature map
        st.markdown("**Original Image:**")
        st.image(sample_img, caption="Input Image", use_container_width=True, clamp=True)
        
        st.markdown("**Feature Map (After Convolution & Activation):**")
        st.image(feature_map, caption="Feature Map", use_container_width=True, clamp=True)
    
    with col2:
        # Pooling options
        pooling_type = st.selectbox(
            "Select Pooling Type:",
            ["Max Pooling", "Average Pooling"]
        )
        
        # Pooling parameters
        pool_size = st.slider("Pool Size:", 2, 8, 2)
        stride = st.slider("Stride:", 1, pool_size, pool_size)
        
        # Apply pooling
        if pooling_type == "Max Pooling":
            pooled_result = apply_pooling(feature_map, "max", pool_size, stride)
            description = f"""
            **Max Pooling (pool_size={pool_size}, stride={stride})**:
            - Takes the **maximum value** from each region
            - Emphasizes the strongest features or activations
            - Preserves sharp features and high activations
            - Most commonly used in modern CNNs
            - Effective for detecting the presence of specific features
            """
        else:  # Average Pooling
            pooled_result = apply_pooling(feature_map, "avg", pool_size, stride)
            description = f"""
            **Average Pooling (pool_size={pool_size}, stride={stride})**:
            - Takes the **average value** from each region
            - Smooths out activations in each pooling window
            - Better preserves background/context information
            - Less common but useful for certain tasks
            - Can help reduce noise in feature maps
            """
        
        # Show pooling visualization
        st.markdown("**After Pooling:**")
        st.image(pooled_result, caption=f"After {pooling_type}", use_container_width=True, clamp=True)
        
        # Show pooling description
        st.markdown(description)
        
        # Show output dimensions
        st.markdown(f"""
        **Input Shape**: {feature_map.shape[0]} × {feature_map.shape[1]}  
        **Output Shape**: {pooled_result.shape[0]} × {pooled_result.shape[1]}  
        **Reduction**: {feature_map.size} → {pooled_result.size} pixels ({100 * pooled_result.size / feature_map.size:.1f}% of original)
        """)
    
    # Visual explanation of pooling
    st.subheader("Visual Explanation of Pooling")
    
    st.markdown("""
    This visualization shows how pooling works on a small section of the feature map.
    The colored grid represents a region of the feature map, and the pooling operation computes a single output value from each window.
    """)
    
    # Create a visual explanation of pooling
    # First, create a simple grid to show the pooling operation
    grid_size = 8
    small_grid = feature_map[:grid_size, :grid_size]
    
    # Create the visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show the input grid
    im = axs[0].imshow(small_grid, cmap='viridis')
    axs[0].set_title("Input Region")
    plt.colorbar(im, ax=axs[0])
    
    # Draw grid lines
    for i in range(grid_size+1):
        axs[0].axhline(i-0.5, color='white', linewidth=0.5)
        axs[0].axvline(i-0.5, color='white', linewidth=0.5)
    
    # Highlight pooling windows
    window_starts = list(range(0, grid_size, pool_size))
    for i in window_starts:
        for j in window_starts:
            rect = plt.Rectangle((j-0.5, i-0.5), pool_size, pool_size, 
                                fill=False, edgecolor='yellow', linewidth=2)
            axs[0].add_patch(rect)
    
    # Add values to cells
    for i in range(grid_size):
        for j in range(grid_size):
            text = axs[0].text(j, i, f"{small_grid[i, j]:.2f}",
                       ha="center", va="center", color="w", fontsize=8)
    
    # Show the pooling operation
    axs[1].axis('off')
    axs[1].set_title("Pooling Operation")
    
    # Draw arrows from windows to results
    pooled_shape = (grid_size + pool_size - 1) // pool_size
    output_grid = np.zeros((pooled_shape, pooled_shape))
    
    mid_x = grid_size / 2
    arrow_length = 2
    
    # Create descriptive text for the pooling operation
    if pooling_type == "Max Pooling":
        pool_op_text = "Max Pooling takes the maximum value from each highlighted window"
    else:
        pool_op_text = "Average Pooling takes the average of all values in each highlighted window"
    
    axs[1].text(0.5, 0.5, pool_op_text, ha='center', va='center', wrap=True, fontsize=12)
    
    # Show the output grid
    axs[2].imshow(pooled_result[:pooled_shape, :pooled_shape], cmap='plasma')
    axs[2].set_title("Output After Pooling")
    
    # Draw grid lines
    for i in range(pooled_shape+1):
        axs[2].axhline(i-0.5, color='white', linewidth=0.5)
        axs[2].axvline(i-0.5, color='white', linewidth=0.5)
    
    # Add values to cells
    for i in range(pooled_shape):
        for j in range(pooled_shape):
            text = axs[2].text(j, i, f"{pooled_result[i, j]:.2f}",
                       ha="center", va="center", color="w", fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 3D visualization to show the effect of pooling
    st.subheader("3D Visualization of Pooling Effect")
    
    x1 = np.arange(0, feature_map.shape[1])
    y1 = np.arange(0, feature_map.shape[0])
    x1_mesh, y1_mesh = np.meshgrid(x1, y1)
    
    x2 = np.arange(0, pooled_result.shape[1])
    y2 = np.arange(0, pooled_result.shape[0])
    x2_mesh, y2_mesh = np.meshgrid(x2, y2)
    
    # Scaling factor to make the pooled result align with the original
    scale_factor = feature_map.shape[0] / pooled_result.shape[0]
    
    # Create the 3D visualization with Plotly
    fig = go.Figure()
    
    # Add the surface for feature map
    fig.add_trace(go.Surface(
        z=feature_map,
        x=x1_mesh, 
        y=y1_mesh,
        colorscale='Viridis',
        opacity=0.8,
        showscale=False,
        name="Before Pooling"
    ))
    
    # Add the upscaled pooled result
    pooled_upscaled = np.repeat(np.repeat(pooled_result, int(scale_factor), axis=0), int(scale_factor), axis=1)
    x_upscaled = np.arange(0, pooled_upscaled.shape[1])
    y_upscaled = np.arange(0, pooled_upscaled.shape[0])
    x_up_mesh, y_up_mesh = np.meshgrid(x_upscaled, y_upscaled)
    
    # Only show the upscaled result to the size of the original
    show_size_x = min(pooled_upscaled.shape[1], feature_map.shape[1])
    show_size_y = min(pooled_upscaled.shape[0], feature_map.shape[0])
    
    fig.add_trace(go.Surface(
        z=pooled_upscaled[:show_size_y, :show_size_x] + 0.05,  # Slight offset for visibility
        x=x_up_mesh[:show_size_y, :show_size_x],
        y=y_up_mesh[:show_size_y, :show_size_x],
        colorscale='Plasma',
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="After Pooling"),
        name="After Pooling"
    ))
    
    fig.update_layout(
        title=f"Effect of {pooling_type} (pool_size={pool_size}, stride={stride})",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Activation Value',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig)
    
    # Comparison of pooling types
    st.subheader("Comparing Pooling Types")
    
    st.markdown("""
    Different pooling types preserve different aspects of the feature maps.
    Here's a comparison of Max Pooling and Average Pooling on the same input:
    """)
    
    # Apply both pooling types
    max_pooled = apply_pooling(feature_map, "max", pool_size, stride)
    avg_pooled = apply_pooling(feature_map, "avg", pool_size, stride)
    
    # Create comparison visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(feature_map, cmap='viridis')
    axs[0].set_title("Original Feature Map")
    axs[0].axis('off')
    
    axs[1].imshow(max_pooled, cmap='plasma')
    axs[1].set_title(f"Max Pooling ({pool_size}×{pool_size}, stride={stride})")
    axs[1].axis('off')
    
    axs[2].imshow(avg_pooled, cmap='plasma')
    axs[2].set_title(f"Average Pooling ({pool_size}×{pool_size}, stride={stride})")
    axs[2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Comparison table
    st.subheader("Max vs. Average Pooling")
    
    comparison_data = {
        "Feature": ["Preserves", "Sensitive to", "Good for", "Common use cases", "Effect on noise"],
        "Max Pooling": ["Strongest features", "Peak activations", "Feature detection", "Most CNN architectures", "Can amplify noise peaks"],
        "Average Pooling": ["Overall activation patterns", "Average intensity", "Texture analysis", "Specialized cases, background modeling", "Smooths out noise"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Practical considerations
    st.subheader("Practical Considerations")
    
    st.markdown("""
    ### Effect of Different Pool Sizes and Strides
    
    - **Larger pool sizes** reduce spatial dimensions more aggressively, but lose more spatial information
    - **Smaller pool sizes** preserve more spatial details, but reduce dimensions less
    - **Stride equal to pool size** is most common (non-overlapping windows)
    - **Stride smaller than pool size** creates overlapping windows, which can improve performance but increases computation
    
    ### Modern Trends
    
    In recent years, some architectures have moved away from explicit pooling:
    - **Strided convolutions** can reduce dimensions similarly to pooling
    - **Global Average Pooling** (taking average of entire feature maps) is popular in modern architectures
    - Some networks use no pooling at all, relying on stride to control dimensions
    
    However, traditional max pooling remains very common and effective in many architectures.
    """)
    
    # Learning checkpoints
    st.subheader("Key Takeaways")
    
    st.markdown("""
    ### What You've Learned:
    
    - **Pooling** reduces spatial dimensions while preserving important information
    - **Max pooling** preserves strongest features and is most commonly used
    - **Average pooling** preserves average activation and can help with texture analysis
    - Pooling provides:
      - Dimensionality reduction (faster computation)
      - Translation invariance (less sensitive to exact feature location)
      - Hierarchical representation (features at different scales)
    
    ### Next Steps:
    
    Now that you understand convolution, activation, and pooling operations, let's explore how they fit together in a complete CNN architecture.
    Proceed to the CNN Architecture Flow tutorial to learn more.
    """)
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("⬅️ [Return to Tutorial Selection](#guided-cnn-tutorials)")
    with col2:
        st.markdown("⬅️ [Previous: Activation Functions](#)")
    with col3:
        st.markdown("➡️ [Next: CNN Architecture Flow](#)")

def architecture_flow_tutorial():
    st.subheader("Understanding CNN Architecture Flow")
    
    st.markdown("""
    ## Data Flow in CNN Architectures
    
    This tutorial visualizes how data flows through a complete CNN architecture, from input to output.
    
    ### Typical CNN Architecture:
    
    The standard pattern in a Convolutional Neural Network is:
    
    1. **Input Layer**: The original image
    2. **Feature Extraction Layers**: Series of Convolution, Activation, and Pooling operations
    3. **Flattening**: Convert 2D feature maps to 1D vector
    4. **Classification Layers**: Fully connected layers for classification or regression
    5. **Output Layer**: Final predictions
    
    Let's trace the flow of data through this architecture:
    """)
    
    # Create a sample flow visualization
    st.markdown("### Interactive CNN Architecture Flow")
    
    # Configure the CNN architecture
    st.markdown("#### Architecture Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_conv_blocks = st.slider("Number of Conv+Pool Blocks:", 1, 3, 2)
    with col2:
        initial_filters = st.slider("Initial Filters:", 4, 64, 16)
    with col3:
        fc_size = st.slider("Dense Layer Size:", 16, 256, 64)
    
    # Basic image input
    input_type = st.selectbox(
        "Select Input Image:",
        ["circle", "edge", "corner", "gradient", "cross"]
    )
    
    sample_img = create_sample_image(input_type, size=32)
    
    # Define flow stages and shapes
    stages = ["Input"]
    shapes = [(1, 32, 32)]  # Starting with a 32x32 grayscale image
    activations = [sample_img]
    
    # Configure filters for convolution demonstration
    filters = []
    filter_names = list(COMMON_FILTERS.keys())
    
    for i in range(num_conv_blocks):
        # Each block has: Conv -> ReLU -> Pooling
        filter_name = filter_names[i % len(filter_names)]
        filters.append(COMMON_FILTERS[filter_name])
        
        # Conv layer
        filter_count = initial_filters * (2 ** i)
        stages.append(f"Conv {i+1}\n({filter_count} filters)")
        
        # Calculate output shape after convolution (using same padding)
        prev_shape = shapes[-1]
        conv_shape = (filter_count, prev_shape[1], prev_shape[2])
        shapes.append(conv_shape)
        
        # Calculate activation for visualization
        conv_result = apply_convolution(activations[-1], filters[-1])
        activations.append(conv_result)
        
        # ReLU layer
        stages.append(f"ReLU {i+1}")
        shapes.append(conv_shape)  # Shape doesn't change
        activations.append(apply_activation(conv_result, "relu"))
        
        # Pooling layer
        stages.append(f"MaxPool {i+1}")
        
        # Calculate output shape after pooling
        pool_shape = (conv_shape[0], conv_shape[1] // 2, conv_shape[2] // 2)
        shapes.append(pool_shape)
        
        # Calculate activation for visualization
        pool_result = apply_pooling(activations[-1], "max", 2, 2)
        activations.append(pool_result)
    
    # Flatten layer
    stages.append("Flatten")
    flat_size = shapes[-1][0] * shapes[-1][1] * shapes[-1][2]
    shapes.append((flat_size,))
    
    # Just reshape the last activation for visualization
    flat_activation = activations[-1].flatten()
    activations.append(flat_activation)
    
    # Fully connected layer
    stages.append(f"Dense\n({fc_size} units)")
    shapes.append((fc_size,))
    
    # Dummy activation for visualization
    fc_activation = np.random.rand(fc_size)
    activations.append(fc_activation)
    
    # Output layer
    num_classes = 10  # Assuming classification task with 10 classes
    stages.append(f"Output\n({num_classes} classes)")
    shapes.append((num_classes,))
    
    # Dummy output for visualization
    output_activation = np.random.rand(num_classes)
    output_activation = output_activation / output_activation.sum()  # Normalize to sum to 1
    activations.append(output_activation)
    
    # Create architecture flow visualization
    st.markdown("#### Network Architecture Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Plot the data flow
    num_stages = len(stages)
    x_positions = np.linspace(0, 100, num_stages)
    y_position = 50
    
    # Draw nodes for each stage
    for i, (stage, shape) in enumerate(zip(stages, shapes)):
        # Calculate node size based on tensor size
        if len(shape) == 3:
            # For 3D tensors, size is proportional to feature maps
            node_size = 5 + 10 * np.log1p(shape[0])
        else:
            # For 1D tensors, size is proportional to length
            node_size = 5 + 10 * np.log1p(shape[0])
        
        # Draw the node
        circle = plt.Circle((x_positions[i], y_position), node_size, 
                           color='lightblue', alpha=0.7)
        ax.add_patch(circle)
        
        # Add text label (stage name and shape)
        if len(shape) == 3:
            shape_text = f"{shape[0]}×{shape[1]}×{shape[2]}"
        else:
            shape_text = f"{shape[0]}"
        
        ax.text(x_positions[i], y_position, f"{stage}\n{shape_text}", 
                ha='center', va='center', fontsize=8)
        
        # Draw connecting lines
        if i > 0:
            ax.plot([x_positions[i-1], x_positions[i]], [y_position, y_position], 'k-')
    
    # Set the axes limits
    ax.set_xlim(-10, 110)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Interactive data flow visualization
    st.markdown("#### Interactive Data Visualization")
    st.markdown("""
    This interactive visualization shows how the data is transformed as it flows through the network.
    Click on different stages to see the corresponding activation visualizations.
    """)
    
    # Select stage to visualize
    selected_stage = st.selectbox("Select Stage to Visualize:", stages)
    selected_idx = stages.index(selected_stage)
    
    # Display the selected activation
    st.markdown(f"**Visualization for: {selected_stage}**")
    
    # Determine how to visualize based on the data shape
    activation = activations[selected_idx]
    shape = shapes[selected_idx]
    
    if selected_idx == 0:
        # Input image
        st.image(activation, caption="Input Image", use_container_width=True, clamp=True)
    
    elif len(shape) == 3:
        # 3D tensor (Conv/Pool layers)
        num_filters_to_show = min(16, shape[0])
        
        if shape[0] == 1:
            # Just one channel
            st.image(activation, caption=f"Activation for {selected_stage}", use_container_width=True, clamp=True)
        
        else:
            # Multiple feature maps - create a grid
            st.markdown(f"**Showing {num_filters_to_show} out of {shape[0]} feature maps**")
            
            # Determine grid layout
            cols = min(4, num_filters_to_show)
            rows = (num_filters_to_show + cols - 1) // cols
            
            fig, axs = plt.subplots(rows, cols, figsize=(12, 3 * rows))
            if rows == 1 and cols == 1:
                axs = np.array([axs])
            axs = axs.flatten()
            
            # Just show a random sample if too many
            filter_indices = list(range(min(16, shape[0])))
            
            for i, idx in enumerate(filter_indices):
                if i < len(axs):
                    # For visualization, just create some fake feature maps
                    # In a real model, these would be the actual activations
                    feature_map = apply_convolution(activation, np.random.randn(3, 3))
                    
                    axs[i].imshow(feature_map, cmap='viridis')
                    axs[i].set_title(f"Filter {idx+1}")
                    axs[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_filters_to_show, len(axs)):
                axs[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Also show a 3D visualization for one feature map
            st.markdown("**3D Visualization of Feature Map 1**")
            
            feature_map = apply_convolution(activation, np.random.randn(3, 3))
            
            x = np.arange(0, feature_map.shape[1])
            y = np.arange(0, feature_map.shape[0])
            x_mesh, y_mesh = np.meshgrid(x, y)
            
            fig = go.Figure(data=[go.Surface(
                z=feature_map,
                x=x_mesh, 
                y=y_mesh,
                colorscale='Viridis',
                contours = {
                    "z": {"show": True, "start": 0, "end": 1, "size": 0.05}
                }
            )])
            
            fig.update_layout(
                title=f"3D View of Feature Map",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Activation',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=700,
                height=500,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig)
    
    else:
        # 1D tensor (FC layers)
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # For output layer, show as probabilities
        if selected_idx == len(stages) - 1:
            bars = ax.bar(range(len(activation)), activation)
            ax.set_title("Output Probabilities")
            ax.set_xticks(range(len(activation)))
            ax.set_xticklabels([f"Class {i}" for i in range(len(activation))])
            ax.set_ylabel("Probability")
            
            # Highlight the predicted class
            pred_class = np.argmax(activation)
            bars[pred_class].set_color('green')
            
        else:
            # For other FC layers, show activations
            ax.bar(range(min(50, len(activation))), activation[:50])
            ax.set_title(f"Activations for {selected_stage} (showing first 50 units)")
            ax.set_xlabel("Neuron")
            ax.set_ylabel("Activation")
        
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Describe the transformation at each stage
    st.subheader("Stage Description")
    
    stage_descriptions = {
        "Input": """
        **Input Layer**
        
        The input layer receives the raw image data, often normalized to the range [0, 1].
        
        - Raw pixel values are arranged in a 3D tensor: (channels, height, width)
        - For grayscale images, there's only one channel
        - For color images, there are typically 3 channels (RGB)
        
        This is the starting point for feature extraction in the CNN.
        """,
        
        "Conv": """
        **Convolutional Layer**
        
        Convolutional layers apply multiple filters to extract different features from the input.
        
        - Each filter produces a feature map highlighting specific patterns
        - Multiple filters capture different aspects (edges, textures, etc.)
        - Outputs a set of feature maps equal to the number of filters
        - Uses parameter sharing to efficiently detect features regardless of location
        
        This layer is responsible for feature extraction from the input.
        """,
        
        "ReLU": """
        **ReLU Activation**
        
        The ReLU (Rectified Linear Unit) activation introduces non-linearity after convolution.
        
        - Replaces all negative values with zeros: f(x) = max(0, x)
        - Keeps positive values unchanged
        - Enables the network to learn complex, non-linear patterns
        - Fast computation and helps prevent vanishing gradient problems
        
        Without activation functions, multiple layers would just perform linear transformations.
        """,
        
        "MaxPool": """
        **Max Pooling Layer**
        
        Max pooling reduces the spatial dimensions while preserving important features.
        
        - Takes the maximum value from each region
        - Reduces height and width, typically by a factor of 2
        - Provides translational invariance (less sensitive to exact feature location)
        - Reduces computation and parameters in subsequent layers
        
        This downsampling helps the network focus on the presence of features rather than their exact location.
        """,
        
        "Flatten": """
        **Flatten Layer**
        
        The flatten layer converts the 3D feature maps to a 1D vector for the fully connected layers.
        
        - Reshapes data from (channels, height, width) to a 1D array
        - No parameters or actual computation, just a reshape operation
        - Connects the convolutional feature extraction to the dense classification layers
        
        This is a transition point from spatial arrangement to feature-based representation.
        """,
        
        "Dense": """
        **Fully Connected (Dense) Layer**
        
        Dense layers connect every input neuron to every output neuron.
        
        - Combines features from all locations to make higher-level abstractions
        - Usually has fewer neurons than the flattened input (dimensionality reduction)
        - Learns complex combinations of features for classification
        - Followed by activation functions (typically ReLU)
        
        These layers perform the actual classification based on extracted features.
        """,
        
        "Output": """
        **Output Layer**
        
        The output layer produces the final predictions.
        
        - For classification, typically has one neuron per class
        - Often uses softmax activation to convert to probabilities (sum to 1)
        - The highest value corresponds to the predicted class
        - For regression, may have a single output with linear activation
        
        This is where the network makes its final decision or prediction.
        """
    }
    
    # Extract the base name (without numbers/subscripts)
    base_stage = re.sub(r'\s*\d+.*', '', selected_stage).strip()
    
    # Display the description
    if base_stage in stage_descriptions:
        st.markdown(stage_descriptions[base_stage])
    else:
        st.markdown(f"Description for {selected_stage} not available.")
    
    # Progressive training
    st.subheader("Progressive Feature Learning")
    
    st.markdown("""
    As a CNN trains, it learns increasingly complex features at each layer:
    
    1. **Early Layers** learn basic features (edges, colors, simple textures)
    2. **Middle Layers** combine these to detect parts and more complex textures
    3. **Deep Layers** detect high-level features and complex patterns
    4. **Fully Connected Layers** combine these features for classification
    
    This hierarchical learning is a key strength of CNNs.
    """)
    
    # Create a visualization of hierarchical feature learning
    st.markdown("#### Hierarchical Feature Learning")
    
    # Show a conceptual diagram
    feature_hierarchy = {
        "Layer 1": ["Edges", "Colors", "Gradients"],
        "Layer 2": ["Textures", "Simple Shapes", "Corners"],
        "Layer 3": ["Parts", "Complex Patterns", "Object Components"],
        "Layer 4": ["Object Features", "Scene Elements", "High-level Concepts"]
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw the hierarchy
    layer_y_positions = np.linspace(0.8, 0.2, len(feature_hierarchy))
    for i, (layer, features) in enumerate(feature_hierarchy.items()):
        y_pos = layer_y_positions[i]
        
        # Draw layer title
        ax.text(0.1, y_pos, layer, fontsize=14, ha='center', va='center', 
                bbox=dict(facecolor='lightblue', alpha=0.5))
        
        # Draw features
        num_features = len(features)
        feature_x = np.linspace(0.3, 0.9, num_features)
        
        for j, feature in enumerate(features):
            ax.text(feature_x[j], y_pos, feature, fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='lightgreen', alpha=0.5))
            
            # Connect to previous layer features
            if i > 0:
                prev_features = feature_hierarchy[list(feature_hierarchy.keys())[i-1]]
                num_prev = len(prev_features)
                prev_x = np.linspace(0.3, 0.9, num_prev)
                prev_y = layer_y_positions[i-1]
                
                # Connect to a couple of previous features
                connections = min(2, num_prev)
                for k in range(connections):
                    ax.plot([feature_x[j], prev_x[j % num_prev]], [y_pos, prev_y], 'k-', alpha=0.3)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Hierarchical Feature Learning in CNNs")
    
    st.pyplot(fig)
    
    # Learning checkpoints
    st.subheader("Key Takeaways")
    
    st.markdown("""
    ### What You've Learned:
    
    - CNNs have a **standard architecture flow**: Conv → Activation → Pooling → ... → Flatten → Dense → Output
    - The **feature extraction** part (Conv+ReLU+Pool) learns hierarchical representations
    - The **classification** part (Dense layers) combines these features for prediction
    - Data dimensionality changes throughout the network:
      - **Height and width** generally decrease through pooling
      - **Channel/filter count** generally increases in deeper conv layers
      - **Flattening** converts 3D feature maps to 1D vectors
    - Each layer type serves a specific purpose in the overall architecture
    
    ### Next Steps:
    
    Now that you understand the architecture, learn about how CNNs are trained.
    Proceed to the CNN Training Concepts tutorial to learn more.
    """)
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("⬅️ [Return to Tutorial Selection](#guided-cnn-tutorials)")
    with col2:
        st.markdown("⬅️ [Previous: Pooling Operations](#)")
    with col3:
        st.markdown("➡️ [Next: CNN Training Concepts](#)")

def training_concepts_tutorial():
    st.subheader("CNN Training Concepts")
    
    st.markdown("""
    ## Training Neural Networks
    
    This tutorial covers the fundamental concepts of training Convolutional Neural Networks.
    
    ### The Training Process:
    
    1. **Forward Pass**: Run data through the network to get predictions
    2. **Loss Calculation**: Compare predictions to actual labels
    3. **Backward Pass**: Calculate gradients of the loss w.r.t. weights
    4. **Weight Update**: Adjust weights to minimize the loss
    
    Let's explore each of these concepts:
    """)
    
    # Loss function explanation
    st.markdown("### Loss Functions")
    
    st.markdown("""
    Loss functions measure how far the network's predictions are from the actual labels.
    Different tasks require different loss functions:
    """)
    
    # Loss function visualization
    col1, col2 = st.columns(2)
    
    with col1:
        loss_type = st.selectbox(
            "Select Loss Function:",
            ["Cross-Entropy Loss", "Mean Squared Error", "Binary Cross-Entropy"]
        )
        
        # Number of classes (for classification losses)
        if loss_type in ["Cross-Entropy Loss", "Binary Cross-Entropy"]:
            num_classes = st.slider("Number of Classes:", 2, 10, 
                                    value=2 if loss_type == "Binary Cross-Entropy" else 5,
                                    disabled=(loss_type == "Binary Cross-Entropy"))
        
    with col2:
        if loss_type == "Cross-Entropy Loss":
            st.markdown("""
            **Cross-Entropy Loss**
            
            Used for multi-class classification problems.
            
            **Formula**: -∑(y_true * log(y_pred))
            
            **Properties**:
            - Higher penalty for confident wrong predictions
            - Good for classification with multiple categories
            - Works well with softmax output
            
            **Common Use Cases**: Image classification, natural language processing
            """)
            
        elif loss_type == "Mean Squared Error":
            st.markdown("""
            **Mean Squared Error (MSE)**
            
            Used for regression problems or when predicting continuous values.
            
            **Formula**: (1/n) * ∑(y_true - y_pred)²
            
            **Properties**:
            - Penalizes larger errors more heavily (squared term)
            - Differentiable everywhere
            - Sensitive to outliers
            
            **Common Use Cases**: Regression, image reconstruction, value prediction
            """)
            
        elif loss_type == "Binary Cross-Entropy":
            st.markdown("""
            **Binary Cross-Entropy**
            
            Used for binary classification problems (two classes).
            
            **Formula**: -[y_true*log(y_pred) + (1-y_true)*log(1-y_pred)]
            
            **Properties**:
            - Special case of cross-entropy for binary problems
            - Works well with sigmoid output
            - Good for imbalanced datasets when properly weighted
            
            **Common Use Cases**: Binary classification, segmentation (per-pixel)
            """)
    
    # Visualization of the loss function
    st.markdown("#### Loss Function Visualization")
    
    if loss_type == "Cross-Entropy Loss":
        # Visualize cross-entropy loss for a few examples
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot: Loss vs predicted probability for correct class
        p = np.linspace(0.01, 1, 100)
        loss = -np.log(p)
        
        axs[0].plot(p, loss)
        axs[0].set_xlabel("Predicted Probability for True Class")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Cross-Entropy Loss vs. Prediction")
        axs[0].grid(True, alpha=0.3)
        
        # Second plot: Example predictions and resulting loss
        # Create some example predictions
        true_class = np.random.randint(0, num_classes)
        
        # A few example predictions with different confidence levels
        predictions = []
        for confidence in [0.3, 0.5, 0.7, 0.9]:
            pred = np.ones(num_classes) * ((1 - confidence) / (num_classes - 1))
            pred[true_class] = confidence
            predictions.append(pred)
        
        # Calculate losses
        losses = []
        for pred in predictions:
            true_one_hot = np.zeros(num_classes)
            true_one_hot[true_class] = 1
            loss_val = -np.sum(true_one_hot * np.log(pred + 1e-10))
            losses.append(loss_val)
        
        # Bar plot of different predictions
        bar_width = 0.15
        index = np.arange(num_classes)
        
        for i, pred in enumerate(predictions):
            axs[1].bar(index + i*bar_width, pred, bar_width, 
                      label=f'Pred {i+1}, Loss: {losses[i]:.2f}')
        
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Predicted Probability')
        axs[1].set_title(f'Example Predictions (True Class: {true_class})')
        axs[1].set_xticks(index + bar_width * (len(predictions) - 1) / 2)
        axs[1].set_xticklabels([f'Class {i}' for i in range(num_classes)])
        axs[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif loss_type == "Mean Squared Error":
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot: MSE as a function of prediction error
        errors = np.linspace(-2, 2, 100)
        mse = errors**2
        
        axs[0].plot(errors, mse)
        axs[0].set_xlabel("Prediction Error (y_true - y_pred)")
        axs[0].set_ylabel("MSE Loss")
        axs[0].set_title("MSE Loss vs. Prediction Error")
        axs[0].grid(True, alpha=0.3)
        
        # Second plot: Example regression predictions
        x = np.linspace(0, 10, 20)
        y_true = 0.5 * x + 2 + np.random.normal(0, 1, len(x))
        
        # Three sets of predictions with different errors
        y_pred_good = 0.48 * x + 2.1  # Good predictions
        y_pred_ok = 0.4 * x + 3  # OK predictions
        y_pred_bad = 0.2 * x + 5  # Bad predictions
        
        # Calculate MSE for each
        mse_good = np.mean((y_true - y_pred_good)**2)
        mse_ok = np.mean((y_true - y_pred_ok)**2)
        mse_bad = np.mean((y_true - y_pred_bad)**2)
        
        axs[1].scatter(x, y_true, color='black', label=f'True Values')
        axs[1].plot(x, y_pred_good, 'g-', label=f'Good Model (MSE: {mse_good:.2f})')
        axs[1].plot(x, y_pred_ok, 'b-', label=f'OK Model (MSE: {mse_ok:.2f})')
        axs[1].plot(x, y_pred_bad, 'r-', label=f'Bad Model (MSE: {mse_bad:.2f})')
        
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].set_title('Regression Example with MSE')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif loss_type == "Binary Cross-Entropy":
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot: Binary cross-entropy loss
        p = np.linspace(0.01, 0.99, 100)
        loss_class1 = -np.log(p)  # When true class is 1
        loss_class0 = -np.log(1-p)  # When true class is 0
        
        axs[0].plot(p, loss_class1, 'b-', label='True Class = 1')
        axs[0].plot(p, loss_class0, 'r-', label='True Class = 0')
        axs[0].set_xlabel("Predicted Probability for Class 1")
        axs[0].set_ylabel("Binary Cross-Entropy Loss")
        axs[0].set_title("BCE Loss vs. Prediction")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Second plot: Example binary classifications
        # Create a dataset with two classes
        np.random.seed(42)
        n_points = 100
        class0_x = np.random.normal(3, 1, n_points)
        class0_y = np.random.normal(3, 1, n_points)
        class1_x = np.random.normal(7, 1, n_points)
        class1_y = np.random.normal(7, 1, n_points)
        
        # Create a simple decision boundary
        x_line = np.linspace(0, 10, 100)
        y_line = 10 - x_line  # Diagonal decision boundary
        
        # Plot the data and decision boundary
        axs[1].scatter(class0_x, class0_y, color='blue', alpha=0.5, label='Class 0')
        axs[1].scatter(class1_x, class1_y, color='red', alpha=0.5, label='Class 1')
        axs[1].plot(x_line, y_line, 'k--', label='Decision Boundary')
        
        axs[1].set_xlabel('Feature 1')
        axs[1].set_ylabel('Feature 2')
        axs[1].set_title('Binary Classification Example')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Optimization algorithms
    st.markdown("### Optimization Algorithms")
    
    st.markdown("""
    Optimization algorithms determine how to update the weights based on the gradients.
    Different optimizers have different properties:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimizer = st.selectbox(
            "Select Optimizer:",
            ["Gradient Descent", "Stochastic Gradient Descent (SGD)", "Adam", "RMSprop"]
        )
        
        learning_rate = st.slider("Learning Rate:", 0.001, 0.1, 0.01, step=0.001)
    
    with col2:
        if optimizer == "Gradient Descent":
            st.markdown("""
            **Gradient Descent**
            
            The classical optimization algorithm that updates weights based on the full dataset.
            
            **Formula**: w = w - learning_rate * gradient
            
            **Properties**:
            - Uses the entire dataset for each update
            - Computationally expensive for large datasets
            - Finds the exact direction of steepest descent
            - Slow but stable convergence
            
            **Note**: Rarely used in deep learning due to computational cost
            """)
            
        elif optimizer == "Stochastic Gradient Descent (SGD)":
            st.markdown("""
            **Stochastic Gradient Descent (SGD)**
            
            Updates weights based on individual examples or mini-batches.
            
            **Formula**: w = w - learning_rate * gradient (computed on batch)
            
            **Properties**:
            - Faster computation than full gradient descent
            - Noisier updates, which can help escape local minima
            - Commonly used with mini-batches (compromise between full batch and single example)
            - May require momentum to stabilize learning
            
            **Common Variant**: SGD with Momentum, which adds a velocity term
            """)
            
        elif optimizer == "Adam":
            st.markdown("""
            **Adam (Adaptive Moment Estimation)**
            
            Combines ideas from momentum and RMSprop for adaptive learning rates.
            
            **Formula**: Complex - maintains both first and second moment estimates
            
            **Properties**:
            - Adaptive learning rates for each parameter
            - Momentum-based, helping to escape saddle points
            - Works well for a wide range of problems
            - Generally converges faster than SGD
            - Currently one of the most popular optimizers
            
            **When to Use**: First choice for many deep learning problems
            """)
            
        elif optimizer == "RMSprop":
            st.markdown("""
            **RMSprop (Root Mean Square Propagation)**
            
            Adapts learning rates based on the average of recent gradient magnitudes.
            
            **Formula**: Divides learning rate by the root of the average squared gradients
            
            **Properties**:
            - Addresses the diminishing learning rates problem in AdaGrad
            - Good performance on non-stationary objectives
            - Suitable for RNNs and some computer vision tasks
            - Maintains per-parameter learning rates
            
            **When to Use**: When Adam is too aggressive or unstable
            """)
    
    # Visualization of optimization algorithms
    st.markdown("#### Optimizer Behavior Visualization")
    
    # Create a simple 2D loss surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Loss function (bowl with local minima)
    Z = 0.1 * (X**2 + Y**2) + np.sin(X) * np.cos(Y)
    
    # Create optimizer trajectories
    # Starting point
    start_x, start_y = -4, 4
    
    # Simulate different optimizer paths
    steps = 50
    
    if optimizer == "Gradient Descent":
        # Simple path straight down the gradient
        path_x = [start_x]
        path_y = [start_y]
        
        current_x, current_y = start_x, start_y
        for _ in range(steps):
            # Compute gradient (approximation)
            grad_x = 0.2 * current_x + np.cos(current_x) * np.cos(current_y)
            grad_y = 0.2 * current_y - np.sin(current_x) * np.sin(current_y)
            
            # Update position
            current_x -= learning_rate * grad_x
            current_y -= learning_rate * grad_y
            
            path_x.append(current_x)
            path_y.append(current_y)
        
    elif optimizer == "Stochastic Gradient Descent (SGD)":
        # Add noise to the gradients
        path_x = [start_x]
        path_y = [start_y]
        
        current_x, current_y = start_x, start_y
        for _ in range(steps):
            # Compute gradient with noise
            grad_x = 0.2 * current_x + np.cos(current_x) * np.cos(current_y)
            grad_y = 0.2 * current_y - np.sin(current_x) * np.sin(current_y)
            
            # Add noise to simulate stochasticity
            grad_x += np.random.normal(0, 0.5)
            grad_y += np.random.normal(0, 0.5)
            
            # Update position
            current_x -= learning_rate * grad_x
            current_y -= learning_rate * grad_y
            
            path_x.append(current_x)
            path_y.append(current_y)
        
    elif optimizer == "Adam":
        # Simulate Adam's adaptive behavior and momentum
        path_x = [start_x]
        path_y = [start_y]
        
        current_x, current_y = start_x, start_y
        m_x, m_y = 0, 0  # First moment (momentum)
        v_x, v_y = 0, 0  # Second moment
        beta1, beta2 = 0.9, 0.999  # Adam parameters
        epsilon = 1e-8
        
        for t in range(1, steps + 1):
            # Compute gradient
            grad_x = 0.2 * current_x + np.cos(current_x) * np.cos(current_y)
            grad_y = 0.2 * current_y - np.sin(current_x) * np.sin(current_y)
            
            # Add slight noise
            grad_x += np.random.normal(0, 0.1)
            grad_y += np.random.normal(0, 0.1)
            
            # Update biased first moment estimate
            m_x = beta1 * m_x + (1 - beta1) * grad_x
            m_y = beta1 * m_y + (1 - beta1) * grad_y
            
            # Update biased second raw moment estimate
            v_x = beta2 * v_x + (1 - beta2) * (grad_x**2)
            v_y = beta2 * v_y + (1 - beta2) * (grad_y**2)
            
            # Bias correction
            m_x_corrected = m_x / (1 - beta1**t)
            m_y_corrected = m_y / (1 - beta1**t)
            v_x_corrected = v_x / (1 - beta2**t)
            v_y_corrected = v_y / (1 - beta2**t)
            
            # Update position
            current_x -= learning_rate * m_x_corrected / (np.sqrt(v_x_corrected) + epsilon)
            current_y -= learning_rate * m_y_corrected / (np.sqrt(v_y_corrected) + epsilon)
            
            path_x.append(current_x)
            path_y.append(current_y)
        
    elif optimizer == "RMSprop":
        # Simulate RMSprop's adaptive learning rates
        path_x = [start_x]
        path_y = [start_y]
        
        current_x, current_y = start_x, start_y
        v_x, v_y = 0, 0  # Moving average of squared gradients
        beta = 0.9  # Decay factor
        epsilon = 1e-8
        
        for _ in range(steps):
            # Compute gradient
            grad_x = 0.2 * current_x + np.cos(current_x) * np.cos(current_y)
            grad_y = 0.2 * current_y - np.sin(current_x) * np.sin(current_y)
            
            # Add slight noise
            grad_x += np.random.normal(0, 0.2)
            grad_y += np.random.normal(0, 0.2)
            
            # Update moving average of squared gradients
            v_x = beta * v_x + (1 - beta) * (grad_x**2)
            v_y = beta * v_y + (1 - beta) * (grad_y**2)
            
            # Update position
            current_x -= learning_rate * grad_x / (np.sqrt(v_x) + epsilon)
            current_y -= learning_rate * grad_y / (np.sqrt(v_y) + epsilon)
            
            path_x.append(current_x)
            path_y.append(current_y)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the loss surface as a contour
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Loss')
    
    # Plot the optimizer path
    ax.plot(path_x, path_y, 'r.-', label=f'{optimizer} Path')
    ax.plot(path_x[0], path_y[0], 'go', label='Start')
    ax.plot(path_x[-1], path_y[-1], 'ro', label='End')
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_title(f'{optimizer} Trajectory (LR={learning_rate})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Overfitting and regularization
    st.markdown("### Overfitting and Regularization")
    
    st.markdown("""
    Overfitting occurs when a model learns the training data too well, including its noise and outliers,
    resulting in poor generalization to new data. Regularization techniques help prevent overfitting:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        regularizer = st.selectbox(
            "Select Regularization Technique:",
            ["No Regularization", "L1 Regularization (Lasso)", "L2 Regularization (Ridge)", "Dropout", "Early Stopping"]
        )
        
        if regularizer in ["L1 Regularization (Lasso)", "L2 Regularization (Ridge)"]:
            reg_strength = st.slider("Regularization Strength:", 0.0, 1.0, 0.1, step=0.05)
        
        if regularizer == "Dropout":
            dropout_rate = st.slider("Dropout Rate:", 0.0, 0.8, 0.5, step=0.1)
    
    with col2:
        if regularizer == "No Regularization":
            st.markdown("""
            **No Regularization**
            
            Training without any constraints on the model parameters.
            
            **Properties**:
            - Model is free to fit training data perfectly
            - Can learn complex patterns in the data
            - High risk of overfitting, especially with limited data
            - Often results in large parameter values
            
            **When to Use**: When you have a lot of data compared to parameters, or when underfitting is a concern
            """)
            
        elif regularizer == "L1 Regularization (Lasso)":
            st.markdown(f"""
            **L1 Regularization (Lasso)**
            
            Adds the sum of absolute values of weights to the loss function.
            
            **Formula**: Loss + {reg_strength} * ∑|w|
            
            **Properties**:
            - Encourages sparse models (many weights become exactly zero)
            - Performs feature selection implicitly
            - Good when you suspect many features are irrelevant
            - Creates more interpretable models
            
            **When to Use**: When you want explicit feature selection or very sparse models
            """)
            
        elif regularizer == "L2 Regularization (Ridge)":
            st.markdown(f"""
            **L2 Regularization (Ridge)**
            
            Adds the sum of squared weights to the loss function.
            
            **Formula**: Loss + {reg_strength} * ∑w²
            
            **Properties**:
            - Penalizes large weights without making them exactly zero
            - Helps when features are correlated
            - Generally more stable than L1 regularization
            - Most common form of regularization in neural networks
            
            **When to Use**: General-purpose regularization; when you want to prevent large weights
            """)
            
        elif regularizer == "Dropout":
            st.markdown(f"""
            **Dropout**
            
            Randomly deactivates a fraction of neurons during training.
            
            **Properties**:
            - Prevents co-adaptation of neurons
            - Creates an effect similar to ensemble learning
            - Particularly effective for deep networks
            - Current dropout rate: {dropout_rate} (meaning {dropout_rate*100}% of neurons are randomly disabled during training)
            
            **When to Use**: For deep networks when other regularization methods are insufficient
            """)
            
        elif regularizer == "Early Stopping":
            st.markdown("""
            **Early Stopping**
            
            Stops training when performance on validation data stops improving.
            
            **Properties**:
            - Simple to implement
            - Computationally efficient (saves training time)
            - Requires a validation set
            - Complementary to other regularization methods
            
            **When to Use**: Almost always; one of the most practical regularization techniques
            """)
    
    # Visualization of overfitting and regularization
    st.markdown("#### Overfitting vs. Regularization Visualization")
    
    # Create a synthetic dataset with noise
    np.random.seed(42)
    x_data = np.linspace(-3, 3, 50)
    y_true = 0.5 * x_data**2 + x_data + 2  # True underlying function
    y_data = y_true + np.random.normal(0, 2, len(x_data))  # Add noise
    
    # Plot the data and fits
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of noisy data
    ax.scatter(x_data, y_data, color='blue', alpha=0.6, label='Training Data')
    
    # True function
    x_smooth = np.linspace(-3, 3, 100)
    y_smooth = 0.5 * x_smooth**2 + x_smooth + 2
    ax.plot(x_smooth, y_smooth, 'g-', label='True Function')
    
    # Models with different levels of complexity/regularization
    if regularizer == "No Regularization":
        # Create an overfit model (high-degree polynomial)
        model_overfit = np.polyfit(x_data, y_data, 15)
        y_overfit = np.polyval(model_overfit, x_smooth)
        ax.plot(x_smooth, y_overfit, 'r-', label='Overfit Model (High Degree Polynomial)')
        
        # Create a balanced model
        model_balanced = np.polyfit(x_data, y_data, 2)
        y_balanced = np.polyval(model_balanced, x_smooth)
        ax.plot(x_smooth, y_balanced, 'y-', label='Balanced Model (Quadratic)')
        
    elif regularizer in ["L1 Regularization (Lasso)", "L2 Regularization (Ridge)"]:
        # Simulate regularized fits with different strengths
        
        # Overfit model (high-degree polynomial)
        model_overfit = np.polyfit(x_data, y_data, 15)
        y_overfit = np.polyval(model_overfit, x_smooth)
        ax.plot(x_smooth, y_overfit, 'r-', label='Overfit Model (No Regularization)')
        
        # Simulate effect of regularization by dampening coefficients
        if regularizer == "L1 Regularization (Lasso)":
            # L1 tends to make small coefficients exactly zero
            reg_model = model_overfit.copy()
            reg_model[np.abs(reg_model) < reg_strength * np.max(np.abs(reg_model))] = 0
        else:  # L2
            # L2 reduces the magnitude of all coefficients
            reg_model = model_overfit * (1 - reg_strength)
        
        y_reg = np.polyval(reg_model, x_smooth)
        ax.plot(x_smooth, y_reg, 'm-', label=f'{regularizer} (strength={reg_strength})')
        
    elif regularizer == "Dropout":
        # Simulate dropout effects
        
        # Overfit model (high-degree polynomial)
        model_overfit = np.polyfit(x_data, y_data, 15)
        y_overfit = np.polyval(model_overfit, x_smooth)
        ax.plot(x_smooth, y_overfit, 'r-', label='Overfit Model (No Dropout)')
        
        # Simulate dropout by averaging multiple "thinned" networks
        ensemble_preds = []
        np.random.seed(123)
        
        for _ in range(10):
            # Create a "thinned" network by zeroing some coefficients
            dropout_mask = np.random.binomial(1, 1-dropout_rate, len(model_overfit))
            thinned_model = model_overfit * dropout_mask
            # Scale the remaining coefficients
            if np.sum(dropout_mask) > 0:
                thinned_model = thinned_model * len(thinned_model) / np.sum(dropout_mask)
            y_thinned = np.polyval(thinned_model, x_smooth)
            ensemble_preds.append(y_thinned)
        
        # Average predictions
        y_dropout = np.mean(ensemble_preds, axis=0)
        ax.plot(x_smooth, y_dropout, 'm-', label=f'With Dropout (rate={dropout_rate})')
        
    elif regularizer == "Early Stopping":
        # Simulate early stopping with different training iterations
        
        # Early iterations (underfitting)
        model_early = np.polyfit(x_data[::5], y_data[::5], 15)  # Use less data
        y_early = np.polyval(model_early, x_smooth)
        
        # Just right (early stopping point)
        model_optimal = np.polyfit(x_data, y_data, 3)  # Cubic fit
        y_optimal = np.polyval(model_optimal, x_smooth)
        
        # Late iterations (overfitting)
        model_overfit = np.polyfit(x_data, y_data, 15)  # High-degree polynomial
        y_overfit = np.polyval(model_overfit, x_smooth)
        
        ax.plot(x_smooth, y_early, 'y-', label='Early Training (Underfitting)')
        ax.plot(x_smooth, y_optimal, 'm-', label='Optimal Stopping Point')
        ax.plot(x_smooth, y_overfit, 'r-', label='Late Training (Overfitting)')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Effect of {regularizer} on Model Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Training curves visualization
    st.markdown("#### Training Curves")
    
    st.markdown("""
    Training curves show how loss and accuracy evolve during training.
    They're essential tools for diagnosing overfitting and determining when to stop training.
    """)
    
    # Simulate training curves
    epochs = np.arange(1, 101)
    
    # Different cases of training curves
    curve_type = st.selectbox(
        "Select Training Curve Scenario:",
        ["Ideal Training", "Overfitting", "Underfitting", "Learning Rate Too High", "Learning Rate Too Low"]
    )
    
    # Generate training curves based on scenario
    if curve_type == "Ideal Training":
        train_loss = 1.5 * np.exp(-0.02 * epochs) + 0.1
        val_loss = 1.7 * np.exp(-0.018 * epochs) + 0.15
        
        train_acc = 1 - 0.9 * np.exp(-0.03 * epochs)
        val_acc = 1 - 1.0 * np.exp(-0.025 * epochs)
        
        description = """
        **Ideal Training**
        
        In this scenario, both training and validation metrics improve smoothly:
        
        - Training loss/accuracy consistently improves
        - Validation loss/accuracy follows a similar trend
        - Small gap between training and validation performance
        - Model generalizes well to unseen data
        
        **What to do**: Continue training until validation metrics plateau, then apply early stopping
        """
        
    elif curve_type == "Overfitting":
        train_loss = 1.5 * np.exp(-0.04 * epochs) + 0.01
        val_loss = 1.7 * np.exp(-0.03 * epochs) + 0.2
        # Add increase in validation loss after some point
        val_loss[50:] = val_loss[50:] + 0.005 * (epochs[50:] - 50) ** 1.5
        
        train_acc = 1 - 0.9 * np.exp(-0.05 * epochs)
        val_acc = 1 - 0.8 * np.exp(-0.02 * epochs)
        # Add decrease in validation accuracy
        val_acc[60:] = val_acc[60:] - 0.001 * (epochs[60:] - 60) ** 1.2
        val_acc = np.clip(val_acc, 0.5, 1.0)
        
        description = """
        **Overfitting**
        
        In this scenario, the model performs increasingly well on training data but worsens on validation data:
        
        - Training loss continues to decrease
        - Validation loss decreases initially, then increases
        - Growing gap between training and validation performance
        - Model is memorizing training data rather than generalizing
        
        **What to do**: Apply regularization (dropout, L1/L2), get more training data, or use early stopping
        """
        
    elif curve_type == "Underfitting":
        train_loss = 1.0 - 0.5 * np.exp(-0.005 * epochs)
        val_loss = 1.2 - 0.5 * np.exp(-0.005 * epochs)
        
        train_acc = 0.2 + 0.3 * np.exp(-np.exp(-0.02 * epochs))
        val_acc = 0.18 + 0.3 * np.exp(-np.exp(-0.02 * epochs))
        
        description = """
        **Underfitting**
        
        In this scenario, the model struggles to learn from the training data:
        
        - Both training and validation metrics improve very slowly
        - Performance plateaus at a suboptimal level
        - Small gap between training and validation performance
        - Model is too simple to capture the underlying patterns
        
        **What to do**: Use a more complex model, add more features, reduce regularization, train longer
        """
        
    elif curve_type == "Learning Rate Too High":
        # Erratic behavior with high learning rate
        base_train_loss = 1.5 * np.exp(-0.02 * epochs) + 0.1
        base_val_loss = 1.7 * np.exp(-0.018 * epochs) + 0.15
        
        # Add oscillations
        oscillation = 0.2 * np.sin(epochs / 2)
        train_loss = base_train_loss + oscillation
        val_loss = base_val_loss + oscillation * 1.5
        
        # Corresponding accuracy curves
        train_acc = 0.5 + 0.4 * np.tanh(0.03 * epochs) - 0.1 * np.sin(epochs / 2)
        val_acc = 0.45 + 0.35 * np.tanh(0.025 * epochs) - 0.15 * np.sin(epochs / 2)
        
        description = """
        **Learning Rate Too High**
        
        In this scenario, the learning rate causes unstable training:
        
        - Erratic changes in loss and accuracy
        - Oscillations instead of smooth improvement
        - Potential jumps or spikes in metrics
        - Model may fail to converge
        
        **What to do**: Reduce learning rate, use learning rate scheduling, try a different optimizer
        """
        
    elif curve_type == "Learning Rate Too Low":
        # Very slow progress
        train_loss = 2.0 - 0.9 * np.tanh(0.005 * epochs)
        val_loss = 2.2 - 0.9 * np.tanh(0.005 * epochs)
        
        train_acc = 0.3 + 0.5 * np.tanh(0.007 * epochs)
        val_acc = 0.25 + 0.5 * np.tanh(0.007 * epochs)
        
        description = """
        **Learning Rate Too Low**
        
        In this scenario, training progresses extremely slowly:
        
        - Very gradual improvement in metrics
        - No signs of overfitting or instability
        - Training would require many more epochs to converge
        - Computationally inefficient
        
        **What to do**: Increase learning rate, use learning rate warmup, try an adaptive optimizer like Adam
        """
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display description
    st.markdown(description)
    
    # Learning checkpoints
    st.subheader("Key Takeaways")
    
    st.markdown("""
    ### What You've Learned:
    
    - **Loss Functions** measure how well the model's predictions match the true values
      - Different tasks require different loss functions (cross-entropy for classification, MSE for regression)
    
    - **Optimizers** determine how weights are updated during training
      - From simple (SGD) to adaptive (Adam) approaches with different properties
      - Learning rate is a critical hyperparameter that affects convergence
    
    - **Regularization** helps prevent overfitting
      - L1/L2 penalties, dropout, early stopping all help in different ways
      - Well-regularized models generalize better to new data
    
    - **Training Dynamics** can be monitored through loss and accuracy curves
      - Good training shows steady improvement in both training and validation metrics
      - Different patterns indicate specific problems (overfitting, learning rate issues)
    
    ### Next Steps:
    
    Test your knowledge with the CNN Quiz to reinforce what you've learned.
    """)
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("⬅️ [Return to Tutorial Selection](#guided-cnn-tutorials)")
    with col2:
        st.markdown("⬅️ [Previous: CNN Architecture Flow](#)")
    with col3:
        st.markdown("➡️ [Next: CNN Quiz](#)")

def cnn_quiz():
    st.subheader("CNN Knowledge Quiz")
    
    st.markdown("""
    ## Test Your Knowledge
    
    This quiz will test your understanding of CNN concepts. Try to answer these questions
    to reinforce what you've learned from the tutorials.
    """)
    
    # Initialize session state for quiz score if not present
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    
    if 'answered_questions' not in st.session_state:
        st.session_state.answered_questions = set()
    
    # Quiz questions
    questions = [
        {
            "id": 1,
            "question": "What is the main purpose of convolution in CNNs?",
            "options": [
                "To reduce the spatial dimensions of the image",
                "To extract features from the input image",
                "To introduce non-linearity in the network",
                "To normalize pixel values"
            ],
            "correct": 1,
            "explanation": "Convolution operations help extract features from the input image by applying filters that detect patterns like edges, textures, and shapes. Pooling (not convolution) reduces spatial dimensions, activation functions add non-linearity, and normalization layers handle pixel value ranges."
        },
        {
            "id": 2,
            "question": "What does the ReLU activation function do?",
            "options": [
                "It squashes values between 0 and 1",
                "It replaces negative values with zeros and keeps positive values unchanged",
                "It normalizes the output of each layer",
                "It applies a weighted sum to the inputs"
            ],
            "correct": 1,
            "explanation": "ReLU (Rectified Linear Unit) applies the function f(x) = max(0, x), replacing all negative values with zeros while keeping positive values unchanged. This introduces non-linearity into the network, which is crucial for learning complex patterns."
        },
        {
            "id": 3,
            "question": "What is the primary benefit of max pooling?",
            "options": [
                "It introduces non-linearity",
                "It adds more parameters to the model",
                "It reduces the spatial dimensions while preserving important features",
                "It normalizes the values in the feature maps"
            ],
            "correct": 2,
            "explanation": "Max pooling reduces the spatial dimensions (width and height) of feature maps while preserving the most important information (maximum values). This reduces computation, provides some translation invariance, and helps prevent overfitting."
        },
        {
            "id": 4,
            "question": "In a CNN architecture, what typically happens after the convolutional and pooling layers?",
            "options": [
                "Another round of convolution with larger filters",
                "The network outputs the final prediction directly",
                "Feature maps are flattened and passed to fully connected layers",
                "Input normalization is applied again"
            ],
            "correct": 2,
            "explanation": "After the feature extraction stages (convolution and pooling), the feature maps are flattened into a 1D vector and then passed through fully connected (dense) layers. These dense layers combine the extracted features to make the final classification or regression prediction."
        },
        {
            "id": 5,
            "question": "What happens if you use a filter with negative values for convolution?",
            "options": [
                "The CNN will not work properly",
                "It can detect different patterns such as edges or transitions",
                "The output will always be negative",
                "It will blur the image"
            ],
            "correct": 1,
            "explanation": "Filters with both positive and negative values are common and useful. For example, edge detection filters have both positive and negative values to detect transitions between light and dark regions. The sign pattern in the filter determines what features it detects."
        },
        {
            "id": 6,
            "question": "Which loss function is most appropriate for multi-class image classification?",
            "options": [
                "Mean Squared Error",
                "Binary Cross-Entropy",
                "Categorical Cross-Entropy",
                "Hinge Loss"
            ],
            "correct": 2,
            "explanation": "Categorical Cross-Entropy (also called Softmax Loss) is the standard loss function for multi-class classification problems, where each image belongs to exactly one of several classes. It measures the difference between the predicted probability distribution and the true label distribution."
        },
        {
            "id": 7,
            "question": "How does L2 regularization help prevent overfitting?",
            "options": [
                "By adding more neurons to the network",
                "By penalizing large weight values in the model",
                "By increasing the learning rate during training",
                "By adding more training examples"
            ],
            "correct": 1,
            "explanation": "L2 regularization adds a penalty term to the loss function that is proportional to the sum of squared weights. This discourages large weight values, leading to a simpler model that is less likely to overfit the training data and more likely to generalize well to new data."
        },
        {
            "id": 8,
            "question": "What is a key difference between L1 and L2 regularization?",
            "options": [
                "L1 encourages sparse solutions (more zero weights) while L2 doesn't",
                "L1 works with CNNs but L2 doesn't",
                "L1 is applied to biases while L2 is applied to weights",
                "L1 is used for regression while L2 is used for classification"
            ],
            "correct": 0,
            "explanation": "L1 regularization (adding the sum of absolute weight values to the loss) tends to produce sparse solutions by driving many weights exactly to zero. L2 regularization (adding the sum of squared weights) shrinks all weights toward zero but rarely makes them exactly zero."
        },
        {
            "id": 9,
            "question": "Why is overfitting a concern in CNN training?",
            "options": [
                "It causes the training to be too slow",
                "It makes the model too simple to learn the task",
                "It means the model memorizes training data but generalizes poorly to new data",
                "It requires too much memory during training"
            ],
            "correct": 2,
            "explanation": "Overfitting occurs when a model learns the training data too well, including noise and outliers. This results in excellent performance on training data but poor performance on new, unseen data. In other words, the model fails to generalize, which is the main goal of machine learning."
        },
        {
            "id": 10,
            "question": "What happens in a CNN if you increase the number of filters in convolutional layers?",
            "options": [
                "The model will always perform better",
                "The model can detect more features but has more parameters to train",
                "The spatial dimensions of the feature maps will increase",
                "The training will converge faster"
            ],
            "correct": 1,
            "explanation": "Increasing the number of filters allows the CNN to detect more types of features, but it also increases the number of parameters that need to be trained. This can make the model more powerful but also more prone to overfitting, especially with limited training data."
        }
    ]
    
    # Shuffle questions but maintain order across refreshes
    np.random.seed(42)
    question_order = np.random.permutation(len(questions))
    
    # Display questions
    for i in question_order:
        q = questions[i]
        
        st.markdown(f"### Question {q['id']}: {q['question']}")
        
        # Check if this question has been answered
        answered = q['id'] in st.session_state.answered_questions
        
        # User answer
        user_answer = st.radio(
            f"Select your answer for Question {q['id']}:",
            q['options'],
            key=f"q_{q['id']}",
            disabled=answered
        )
        
        # Submit button
        if not answered:
            if st.button(f"Submit Answer for Question {q['id']}", key=f"submit_{q['id']}"):
                # Check answer
                selected_index = q['options'].index(user_answer)
                if selected_index == q['correct']:
                    st.success("✅ Correct! Well done!")
                    st.session_state.quiz_score += 1
                else:
                    st.error("❌ Incorrect. Let's learn from this.")
                
                # Show explanation
                st.info(f"**Explanation**: {q['explanation']}")
                
                # Mark question as answered
                st.session_state.answered_questions.add(q['id'])
        else:
            # If already answered, show the explanation
            selected_index = q['options'].index(user_answer)
            if selected_index == q['correct']:
                st.success("✅ Correct! Well done!")
            else:
                st.error("❌ Incorrect. Let's learn from this.")
            
            st.info(f"**Explanation**: {q['explanation']}")
        
        st.markdown("---")
    
    # Show overall score if all questions answered
    if len(st.session_state.answered_questions) == len(questions):
        st.subheader("Quiz Completed!")
        
        score_percentage = (st.session_state.quiz_score / len(questions)) * 100
        
        st.markdown(f"### Your Score: {st.session_state.quiz_score}/{len(questions)} ({score_percentage:.1f}%)")
        
        if score_percentage >= 90:
            st.balloons()
            st.success("🏆 Excellent! You have a strong understanding of CNN concepts!")
        elif score_percentage >= 70:
            st.success("👍 Good job! You're getting the hang of CNNs!")
        else:
            st.warning("📚 Keep learning! Review the tutorials to strengthen your understanding.")
        
        # Option to reset quiz
        if st.button("Reset Quiz"):
            st.session_state.quiz_score = 0
            st.session_state.answered_questions = set()
            st.experimental_rerun()
    
    # Display current score
    else:
        st.info(f"Current Score: {st.session_state.quiz_score}/{len(st.session_state.answered_questions)} answered")
    
    # Navigation button
    st.markdown("⬅️ [Return to Tutorial Selection](#guided-cnn-tutorials)")

if __name__ == "__main__":
    guided_tutorials()