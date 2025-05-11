import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from PIL import Image
import cv2
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from io import BytesIO

# Define common filters
COMMON_FILTERS = {
    "Horizontal Edge Detection": np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ]),
    "Vertical Edge Detection": np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]),
    "Sobel X": np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]),
    "Sobel Y": np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]),
    "Sharpen": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    "Gaussian Blur": np.array([
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
    ]),
    "Emboss": np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]),
    "Box Blur": np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]),
    "Laplacian": np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]),
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
}

def apply_convolution(image, kernel):
    """Apply convolution with a given kernel to the image"""
    # Convert image to float for processing
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        # Convert to grayscale
        if isinstance(image, np.ndarray):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = np.array(Image.fromarray(np.uint8(image)).convert('L'))
    else:  # Already grayscale
        gray_image = image.copy().astype(float)
    
    # Get dimensions
    kernel_h, kernel_w = kernel.shape
    image_h, image_w = gray_image.shape
    
    # Calculate padding
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # Create output image
    output = np.zeros_like(gray_image, dtype=float)
    
    # Apply convolution
    for i in range(pad_h, image_h - pad_h):
        for j in range(pad_w, image_w - pad_w):
            # Extract the region of interest
            region = gray_image[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            # Apply the kernel
            output[i, j] = np.sum(region * kernel)
    
    # Normalize output to [0, 1] range
    if np.max(output) != np.min(output):
        output = (output - np.min(output)) / (np.max(output) - np.min(output))
    else:
        output = np.zeros_like(output)
    
    return output

def apply_activation(feature_map, activation_name):
    """Apply activation function to feature map"""
    if activation_name == "ReLU":
        return np.maximum(0, feature_map)
    elif activation_name == "Sigmoid":
        return 1 / (1 + np.exp(-feature_map))
    elif activation_name == "Tanh":
        return np.tanh(feature_map)
    else:
        return feature_map

def max_pooling(feature_map, pool_size=2):
    """Apply max pooling operation"""
    h, w = feature_map.shape
    pooled_h, pooled_w = h // pool_size, w // pool_size
    pooled = np.zeros((pooled_h, pooled_w))
    
    for i in range(pooled_h):
        for j in range(pooled_w):
            pooled[i, j] = np.max(feature_map[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    
    return pooled

def avg_pooling(feature_map, pool_size=2):
    """Apply average pooling operation"""
    h, w = feature_map.shape
    pooled_h, pooled_w = h // pool_size, w // pool_size
    pooled = np.zeros((pooled_h, pooled_w))
    
    for i in range(pooled_h):
        for j in range(pooled_w):
            pooled[i, j] = np.mean(feature_map[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    
    return pooled

def create_sample_image(type="circle", size=32):
    """Create a sample image for demonstration"""
    img = np.zeros((size, size))
    
    if type == "circle":
        # Create a circle
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        img[mask] = 1
    
    elif type == "edge":
        # Create a simple edge
        img[:, size//2:] = 1
    
    elif type == "corner":
        # Create a corner
        img[size//2:, size//2:] = 1
    
    elif type == "gradient":
        # Create a gradient
        x = np.linspace(0, 1, size)
        img = np.tile(x, (size, 1))
    
    elif type == "cross":
        # Create a cross pattern
        img[size//2-size//8:size//2+size//8, :] = 1
        img[:, size//2-size//8:size//2+size//8] = 1
    
    return img

def animate_convolution(image, kernel, save_animation=False):
    """Animate the convolution process step by step, showing the sliding window"""
    # Convert image to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    # Convert to float and normalize
    gray_image = gray_image.astype(float) / 255.0
    
    # Get dimensions
    kernel_h, kernel_w = kernel.shape
    image_h, image_w = gray_image.shape
    
    # Calculate padding
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # Initialize the output image
    output = np.zeros_like(gray_image)
    
    # Create figure and axes for animation
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set titles
    ax[0].set_title('Input Image with Filter Window')
    ax[1].set_title('Filter Values')
    ax[2].set_title('Output Feature Map')
    
    # Display the input image
    im1 = ax[0].imshow(gray_image, cmap='gray')
    
    # Create a rectangle patch for the filter window
    from matplotlib import patches
    rect = patches.Rectangle((0, 0), kernel_w, kernel_h, linewidth=2, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    
    # Display the filter
    im2 = ax[1].imshow(kernel, cmap='viridis')
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    
    # Display the empty output image
    im3 = ax[2].imshow(output, cmap='viridis')
    plt.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)
    
    # Turn off axis
    for a in ax:
        a.axis('off')
    
    # Create a list to store frames for saving the animation if needed
    frames = []
    
    # Function to update the animation
    def update(frame):
        # Calculate the current position in the image
        total_steps = (image_h - kernel_h + 1) * (image_w - kernel_w + 1)
        i = frame // (image_w - kernel_w + 1) + pad_h
        j = frame % (image_w - kernel_w + 1) + pad_w
        
        # Extract the region of interest
        region = gray_image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
        
        # Calculate the convolution result
        result = np.sum(region * kernel)
        
        # Update the output image
        output[i, j] = result
        
        # Update the rectangle position on the input image
        rect.set_xy((j-pad_w, i-pad_h))
        
        # Display the current region and kernel
        im1.set_array(gray_image)
        
        # Highlight the current region being processed
        highlighted_image = gray_image.copy()
        highlighted_image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1] = region * 1.5  # Brighten
        im1.set_array(highlighted_image)
        
        # Display the output
        normalized_output = output.copy()
        if np.max(normalized_output) != np.min(normalized_output):
            normalized_output = (normalized_output - np.min(normalized_output)) / (np.max(normalized_output) - np.min(normalized_output))
        im3.set_array(normalized_output)
        
        # Update title to show current position
        ax[0].set_title(f'Input Image (Position: {i-pad_h},{j-pad_w})')
        ax[2].set_title(f'Output Feature Map (Value: {result:.2f})')
        
        # No need to store frames for saving - we'll save directly
        
        return [im1, rect, im3]
    
    # Create the animation
    total_frames = (image_h - 2*pad_h) * (image_w - 2*pad_w)
    anim = FuncAnimation(fig, update, frames=range(total_frames), interval=50, blit=False)
    
    # Save animation as GIF if requested
    if save_animation:
        try:
            anim.save('convolution_animation.gif', writer='pillow', fps=5)
        except Exception as e:
            st.warning(f"Couldn't save animation: {e}")
    
    # Return the animation
    return anim, fig

def apply_pooling(feature_map, pooling_name, pool_size=2):
    """Apply selected pooling operation"""
    if pooling_name == "Max Pooling":
        return max_pooling(feature_map, pool_size)
    elif pooling_name == "Average Pooling":
        return avg_pooling(feature_map, pool_size)
    else:
        return feature_map

def compare_filters_side_by_side(image, filter_names, activation_name="ReLU"):
    """Compare different filters side by side on the same image"""
    # Make sure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale
        if isinstance(image, np.ndarray):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = np.array(Image.fromarray(np.uint8(image)).convert('L'))
    else:
        gray_image = image.copy()
    
    # Create a dictionary to store results
    results = {}
    
    # Apply each filter
    for filter_name in filter_names:
        kernel = COMMON_FILTERS[filter_name]
        # Apply convolution
        conv_result = apply_convolution(gray_image, kernel)
        # Apply activation
        activation_result = apply_activation(conv_result, activation_name)
        # Store results
        results[filter_name] = {
            'kernel': kernel,
            'conv_result': conv_result,
            'activation_result': activation_result
        }
    
    return results

def visualize_receptive_field(image_size=64, layers=3, filter_size=3, stride=1, pooling_size=2):
    """Visualize how the receptive field grows as data passes through the network"""
    # Calculate the network architecture
    layer_configs = []
    
    # Input layer
    input_size = image_size
    rf_size = 1  # Receptive field size starts at 1 (a single pixel)
    layer_configs.append({
        'name': 'Input',
        'size': input_size,
        'rf_size': rf_size,
        'stride': 1
    })
    
    # Add convolutional and pooling layers
    current_size = input_size
    current_stride = 1
    
    for i in range(layers):
        # Convolutional layer
        current_size = current_size - filter_size + 1  # Size after convolution
        rf_size = rf_size + (filter_size - 1) * current_stride  # Receptive field grows
        
        layer_configs.append({
            'name': f'Conv{i+1}',
            'size': current_size,
            'rf_size': rf_size,
            'stride': current_stride
        })
        
        # Pooling layer (if not the last layer)
        if i < layers - 1:
            current_size = current_size // pooling_size  # Size after pooling
            current_stride = current_stride * pooling_size  # Effective stride increases
            
            layer_configs.append({
                'name': f'Pool{i+1}',
                'size': current_size,
                'rf_size': rf_size,  # RF size doesn't change, but effective stride does
                'stride': current_stride
            })
    
    return layer_configs

def create_backprop_visualization(layers=3, neurons_per_layer=[5, 4, 3, 2], input_size=5):
    """Create a simplified visualization of backpropagation through a neural network"""
    # Placeholder for a simple neural network
    weights = []
    activations = []
    gradients = []
    
    # Generate random weights, activations, and gradients
    np.random.seed(42)  # For reproducibility
    
    # Neural network with random weights and activations
    for i in range(len(neurons_per_layer) - 1):
        weights.append(np.random.randn(neurons_per_layer[i], neurons_per_layer[i+1]))
        activations.append(np.random.rand(neurons_per_layer[i]))
    
    # Add final layer activation
    activations.append(np.random.rand(neurons_per_layer[-1]))
    
    # Set output gradients (assuming MSE loss)
    output_gradients = np.random.randn(neurons_per_layer[-1])
    
    # Compute gradients for each layer
    layer_gradients = [output_gradients]
    for i in range(len(weights) - 1, -1, -1):
        grad = np.dot(layer_gradients[0], weights[i].T)
        layer_gradients.insert(0, grad)
    
    # Now layer_gradients[0] is for the input layer, etc.
    
    return {
        'weights': weights,
        'activations': activations,
        'gradients': layer_gradients,
        'neurons_per_layer': neurons_per_layer
    }

def create_feature_heatmap(image, conv_result):
    """Create a heatmap overlay to show which regions of the original image contributed most to activations"""
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Use the RGB image as is
        rgb_image = image.copy()
    else:
        # Convert grayscale to RGB
        rgb_image = np.stack([image] * 3, axis=2)
    
    # Resize conv_result to match image dimensions if necessary
    if conv_result.shape != image.shape[:2]:
        # Simple scaling
        h_ratio = image.shape[0] / conv_result.shape[0]
        w_ratio = image.shape[1] / conv_result.shape[1]
        
        scaled_conv = np.zeros(image.shape[:2])
        
        for i in range(conv_result.shape[0]):
            for j in range(conv_result.shape[1]):
                h_start = int(i * h_ratio)
                h_end = int((i + 1) * h_ratio)
                w_start = int(j * w_ratio)
                w_end = int((j + 1) * w_ratio)
                
                scaled_conv[h_start:h_end, w_start:w_end] = conv_result[i, j]
        
        conv_result = scaled_conv
    
    # Normalize the conv_result to [0, 1]
    if np.max(conv_result) != np.min(conv_result):
        normalized_conv = (conv_result - np.min(conv_result)) / (np.max(conv_result) - np.min(conv_result))
    else:
        normalized_conv = np.zeros_like(conv_result)
    
    # Create a heatmap
    heatmap = np.zeros((*normalized_conv.shape, 3))
    
    # Red channel for high activations
    heatmap[:, :, 0] = normalized_conv
    
    # Create a blended image (original + heatmap)
    alpha = 0.5  # Transparency factor
    blended = rgb_image.astype(float) * (1 - alpha) + heatmap * 255 * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended, heatmap * 255

# Function 'interactive_filter_creator' removed as it's no longer needed

def advanced_visualizations():
    st.title("Advanced CNN Visualizations")
    
    tabs = st.tabs([
        "1. Convolution Animation", 
        "2. Filter Comparison", 
        "3. Backpropagation",
        "4. Feature Highlighting",
        "5. Receptive Field"
    ])
    
    # Tab 1: Convolution Animation
    with tabs[0]:
        st.header("Convolution Animation")
        st.write("""
        This visualization shows how the convolution operation works by animating the sliding window 
        as it moves across the input image. Watch how each value in the output feature map is calculated.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input image selection
            image_type = st.selectbox(
                "Select Input Image:",
                ["circle", "edge", "corner", "gradient", "cross"],
                key="anim_img_type"
            )
            
            # Image size
            image_size = st.slider(
                "Image Size:",
                min_value=16,
                max_value=64,
                value=32,
                step=8,
                key="anim_img_size"
            )
            
            # Create the sample image
            sample_img = create_sample_image(image_type, size=image_size)
            
            # Display the sample image
            st.image(sample_img, caption="Sample Image", use_container_width=True, clamp=True)
        
        with col2:
            # Filter selection
            filter_name = st.selectbox(
                "Select Filter:",
                list(COMMON_FILTERS.keys()),
                key="anim_filter"
            )
            
            # Animation speed
            animation_speed = st.slider(
                "Animation Speed:",
                min_value=1,
                max_value=10,
                value=5,
                key="anim_speed"
            )
            
            # Selected filter
            selected_filter = COMMON_FILTERS[filter_name]
            
            # Display the filter
            st.write(f"**{filter_name} Filter:**")
            
            # Create a heatmap visualization of the filter
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
        
        # Generate and display the animation
        if st.button("Generate Convolution Animation", key="gen_anim_btn"):
            with st.spinner("Generating animation... This may take a moment."):
                # Save animation to GIF
                anim, fig = animate_convolution(sample_img, selected_filter, save_animation=True)
                
                # Display the saved animation
                try:
                    st.image("convolution_animation.gif", caption="Convolution Animation", use_container_width=True)
                    st.success("Animation generated successfully!")
                except:
                    st.error("Could not generate animation. Please try with a smaller image size.")
                    st.pyplot(fig)  # Show the static figure instead
        
        st.write("""
        ### Understanding the Animation
        
        - The **left image** shows the input with the current filter position highlighted
        - The **middle image** shows the filter values
        - The **right image** shows the output feature map being built
        
        The animation demonstrates how each position in the output feature map is calculated by
        multiplying the filter with the corresponding region in the input image and summing the results.
        """)
    
    # Tab 2: Filter Comparison
    with tabs[1]:
        st.header("Side-by-Side Filter Comparison")
        st.write("""
        Compare how different filters respond to the same input image. This helps understand
        what types of features each filter is designed to detect.
        """)
        
        # Image selection
        col1, col2 = st.columns(2)
        
        with col1:
            image_type = st.selectbox(
                "Select Sample Image:",
                ["circle", "edge", "corner", "gradient", "cross"],
                key="compare_img_type"
            )
            
            # Image size
            image_size = st.slider(
                "Image Size:",
                min_value=32,
                max_value=128,
                value=64,
                step=16,
                key="compare_img_size"
            )
            
            # Create the sample image
            sample_img = create_sample_image(image_type, size=image_size)
            
            # Display the sample image
            st.image(sample_img, caption="Sample Image", use_container_width=True, clamp=True)
        
        with col2:
            # Filter selection (multi-select)
            selected_filter_names = st.multiselect(
                "Select Filters to Compare:",
                list(COMMON_FILTERS.keys()),
                default=["Horizontal Edge Detection", "Vertical Edge Detection", "Sobel X"],
                key="compare_filters"
            )
            
            # Activation function
            activation_name = st.selectbox(
                "Activation Function:",
                ["ReLU", "Sigmoid", "Tanh"],
                key="compare_activation"
            )
            
            # Ensure at least one filter is selected
            if not selected_filter_names:
                st.warning("Please select at least one filter to compare.")
                selected_filter_names = ["Horizontal Edge Detection"]
        
        # Compare filters
        if len(selected_filter_names) > 0:
            # Apply filters
            results = compare_filters_side_by_side(sample_img, selected_filter_names, activation_name)
            
            # Display results
            st.subheader("Comparison Results")
            
            # Organize by filter
            num_filters = len(selected_filter_names)
            
            # Show filter kernels
            st.write("### Filter Kernels")
            cols = st.columns(min(num_filters, 3))
            
            for i, filter_name in enumerate(selected_filter_names):
                with cols[i % 3]:
                    kernel = results[filter_name]['kernel']
                    
                    # Display kernel as heatmap
                    fig, ax = plt.subplots(figsize=(3, 3))
                    im = ax.imshow(kernel, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    
                    # Add kernel values as text
                    for i in range(kernel.shape[0]):
                        for j in range(kernel.shape[1]):
                            text = ax.text(j, i, f"{kernel[i, j]:.1f}",
                                   ha="center", va="center", color="w" if abs(kernel[i, j]) > 0.5 else "k")
                    
                    ax.set_title(filter_name)
                    st.pyplot(fig)
            
            # Show convolution results
            st.write("### Convolution Results")
            cols = st.columns(min(num_filters, 3))
            
            for i, filter_name in enumerate(selected_filter_names):
                with cols[i % 3]:
                    conv_result = results[filter_name]['conv_result']
                    st.write(f"**{filter_name}**")
                    st.image(conv_result, caption=f"Conv with {filter_name}", use_container_width=True, clamp=True)
            
            # Show activation results
            st.write(f"### After {activation_name} Activation")
            cols = st.columns(min(num_filters, 3))
            
            for i, filter_name in enumerate(selected_filter_names):
                with cols[i % 3]:
                    activation_result = results[filter_name]['activation_result']
                    st.write(f"**{filter_name}**")
                    st.image(activation_result, caption=f"{filter_name} + {activation_name}", use_container_width=True, clamp=True)
            
            # 3D visualization comparison
            st.subheader("3D Feature Map Comparison")
            
            # Select which visualization to show
            visualization_type = st.radio(
                "Select visualization type:",
                ["After Convolution", f"After {activation_name} Activation"],
                horizontal=True,
                key="compare_viz_type"
            )
            
            # Performance optimization option
            downsample = st.checkbox("Downsample for faster rendering", value=True, key="compare_downsample")
            
            if downsample:
                downsample_factor = st.slider("Downsample factor", min_value=1, max_value=8, value=2, key="compare_downsample_factor")
            else:
                downsample_factor = 1
            
            # Create 3D plots
            num_cols = min(2, num_filters)
            cols = st.columns(num_cols)
            
            for i, filter_name in enumerate(selected_filter_names):
                with cols[i % num_cols]:
                    # Get the feature map based on visualization type
                    if visualization_type == "After Convolution":
                        feature_map = results[filter_name]['conv_result']
                        title = f"{filter_name} Convolution"
                    else:  # After Activation
                        feature_map = results[filter_name]['activation_result']
                        title = f"{filter_name} + {activation_name}"
                    
                    # Downsample if requested
                    if downsample and downsample_factor > 1:
                        h, w = feature_map.shape
                        new_h, new_w = h // downsample_factor, w // downsample_factor
                        
                        if new_h >= 10 and new_w >= 10:
                            feature_map = feature_map[::downsample_factor, ::downsample_factor]
                    
                    # Create 3D plot
                    with st.spinner(f"Generating 3D visualization for {filter_name}..."):
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
                            title=title,
                            scene=dict(
                                xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Activation',
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                            ),
                            width=400,
                            height=400,
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Backpropagation Visualization
    with tabs[2]:
        st.header("Simplified Backpropagation Visualization")
        st.write("""
        This visualization provides a simplified view of how gradients flow backward through a neural network
        during training, which is essential for updating weights and learning from data.
        """)
        
        # Network configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Number of layers
            num_layers = st.slider(
                "Number of Layers:",
                min_value=2,
                max_value=5,
                value=3,
                step=1,
                key="backprop_layers"
            )
            
            # Number of neurons per layer
            neuron_config = []
            neuron_config.append(st.slider("Input Layer Neurons:", min_value=2, max_value=10, value=5, key="input_neurons"))
            
            for i in range(num_layers-1):
                neuron_config.append(st.slider(f"Hidden Layer {i+1} Neurons:", min_value=2, max_value=8, value=5-i, key=f"hidden{i}_neurons"))
            
            neuron_config.append(st.slider("Output Layer Neurons:", min_value=1, max_value=5, value=2, key="output_neurons"))
        
        with col2:
            # Visualization direction
            direction = st.radio(
                "Direction to Visualize:",
                ["Forward Pass", "Backward Pass (Gradients)", "Both"],
                key="backprop_direction"
            )
            
            # Animation speed
            animation_speed = st.slider(
                "Animation Speed:",
                min_value=1,
                max_value=10,
                value=5,
                key="backprop_speed"
            )
            
            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme:",
                ["viridis", "plasma", "inferno", "magma", "cividis"],
                key="backprop_colors"
            )
        
        # Generate network visualization
        network_data = create_backprop_visualization(
            layers=num_layers, 
            neurons_per_layer=neuron_config
        )
        
        # Display the visualization
        st.subheader("Neural Network Visualization")
        
        # Create a figure
        figsize = (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set background color
        ax.set_facecolor('#f0f0f0')
        
        # Network dimensions
        layer_spacing = 4
        neuron_spacing = 2
        max_neurons = max(network_data['neurons_per_layer'])
        
        # Draw neurons
        neuron_positions = []
        for l, n_count in enumerate(network_data['neurons_per_layer']):
            layer_pos = []
            for n in range(n_count):
                # Calculate neuron position
                x = l * layer_spacing
                y = (max_neurons - n_count) / 2 + n * neuron_spacing
                
                # Store position
                layer_pos.append((x, y))
                
                # Draw neuron
                circle = plt.Circle((x, y), 0.8, fill=True, 
                                  color=plt.cm.get_cmap(color_scheme)(network_data['activations'][l][n]))
                ax.add_patch(circle)
                
                # Add activation value
                ax.text(x, y, f"{network_data['activations'][l][n]:.2f}", 
                       ha='center', va='center', color='white', fontsize=9)
            
            neuron_positions.append(layer_pos)
        
        # Draw connections (weights)
        for l in range(len(network_data['weights'])):
            for i in range(network_data['neurons_per_layer'][l]):
                for j in range(network_data['neurons_per_layer'][l+1]):
                    # Get positions
                    x1, y1 = neuron_positions[l][i]
                    x2, y2 = neuron_positions[l+1][j]
                    
                    # Calculate connection strength (normalized weight)
                    weight = network_data['weights'][l][i, j]
                    normalized_weight = (weight - np.min(network_data['weights'][l])) / \
                                       (np.max(network_data['weights'][l]) - np.min(network_data['weights'][l]) + 1e-6)
                    
                    # Draw line with thickness proportional to weight
                    line = plt.Line2D([x1, x2], [y1, y2], 
                                     color='black', 
                                     alpha=0.6,
                                     linewidth=1+2*normalized_weight)
                    ax.add_line(line)
                    
                    # Add small annotation for weight value
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{weight:.1f}", 
                           ha='center', va='center', fontsize=7,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
        # If showing gradients
        if direction in ["Backward Pass (Gradients)", "Both"]:
            # Draw gradient flow
            for l in range(len(network_data['neurons_per_layer'])):
                for n in range(network_data['neurons_per_layer'][l]):
                    # Draw gradient value next to neuron
                    x, y = neuron_positions[l][n]
                    gradient = network_data['gradients'][l][n]
                    
                    # Add gradient label
                    ax.text(x, y-1.2, f"∇:{gradient:.2f}", 
                           ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.9))
        
        # Set axis limits
        ax.set_xlim(-1, (len(network_data['neurons_per_layer']) - 1) * layer_spacing + 1)
        ax.set_ylim(-1, (max_neurons + 1) * neuron_spacing)
        
        # Add labels
        ax.set_title("Neural Network with Weights and Gradients", fontsize=16)
        
        # Add layer labels
        for l, n_count in enumerate(network_data['neurons_per_layer']):
            layer_label = "Input" if l == 0 else "Output" if l == len(network_data['neurons_per_layer']) - 1 else f"Hidden {l}"
            ax.text(l * layer_spacing, -0.5, layer_label, ha='center', fontsize=12)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a legend
        import matplotlib.patches as mpatches
        
        legend_elements = [
            mpatches.Patch(color=plt.cm.get_cmap(color_scheme)(0.2), label='Low Activation'),
            mpatches.Patch(color=plt.cm.get_cmap(color_scheme)(0.8), label='High Activation'),
            plt.Line2D([0], [0], color='black', linewidth=1, label='Weak Connection'),
            plt.Line2D([0], [0], color='black', linewidth=3, label='Strong Connection'),
        ]
        
        if direction in ["Backward Pass (Gradients)", "Both"]:
            legend_elements.append(
                mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Gradient (∇)')
            )
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Show the plot
        st.pyplot(fig)
        
        # Add explanation
        st.write("""
        ### Understanding Backpropagation
        
        Backpropagation is the algorithm that allows neural networks to learn from their mistakes:
        
        1. **Forward Pass**: The input data flows through the network, layer by layer, 
           producing an output prediction.
        
        2. **Error Calculation**: The difference between the predicted output and the actual target 
           value is calculated as the error.
        
        3. **Backward Pass**: The error is propagated backward through the network, 
           computing how much each weight contributed to the error.
        
        4. **Weight Update**: The weights are adjusted to reduce the error, typically using 
           gradient descent.
        
        In the visualization:
        - **Circle color** represents activation strength
        - **Line thickness** represents weight magnitude
        - **Yellow boxes** show the gradient values (∇) flowing backward
        
        During backpropagation, gradients tend to be larger for weights that contributed more to the error,
        guiding the network to make more significant adjustments to these weights.
        """)
    
    # Tab 4: Feature Highlighting (now tab 4 since we removed the filter creator)
    with tabs[3]:
        st.header("Feature Highlighting")
        st.write("""
        This visualization shows which regions of the original image contribute most to activations
        in the feature maps. It helps understand what the CNN is "looking at" when processing an image.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Image selection
            image_type = st.selectbox(
                "Select Sample Image:",
                ["circle", "edge", "corner", "gradient", "cross"],
                key="highlight_img_type"
            )
            
            # Image size
            image_size = st.slider(
                "Image Size:",
                min_value=32,
                max_value=128,
                value=64,
                step=16,
                key="highlight_img_size"
            )
            
            # Create the sample image
            sample_img = create_sample_image(image_type, size=image_size)
            
            # Display the sample image
            st.image(sample_img, caption="Original Image", use_container_width=True, clamp=True)
        
        with col2:
            # Filter selection
            filter_name = st.selectbox(
                "Select Filter:",
                list(COMMON_FILTERS.keys()),
                key="highlight_filter"
            )
            
            # Selected filter
            selected_filter = COMMON_FILTERS[filter_name]
            
            # Display the filter
            st.write(f"**{filter_name} Filter:**")
            
            # Create a heatmap visualization of the filter
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
        
        # Apply convolution
        conv_result = apply_convolution(sample_img, selected_filter)
        
        # Create feature heatmap
        blended, heatmap = create_feature_heatmap(
            # Convert grayscale to RGB for blending
            np.stack([sample_img] * 3, axis=2) if len(sample_img.shape) == 2 else sample_img,
            conv_result
        )
        
        # Display results
        st.subheader("Feature Activation Heatmap")
        
        # Show side by side
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Original Image**")
            if len(sample_img.shape) == 2:
                # Convert grayscale to RGB for consistent display
                display_img = np.stack([sample_img] * 3, axis=2) * 255
                st.image(display_img.astype(np.uint8), use_container_width=True, clamp=True)
            else:
                st.image(sample_img, use_container_width=True, clamp=True)
        
        with col2:
            st.write("**Activation Map**")
            st.image(conv_result, use_container_width=True, clamp=True)
        
        with col3:
            st.write("**Highlighted Features**")
            st.image(blended, use_container_width=True, clamp=True)
        
        st.write("""
        ### Understanding Feature Highlighting
        
        The heatmap overlay shows which parts of the original image produced the strongest activations:
        
        - **Red regions**: Areas with high activation (strong filter response)
        - **Dark regions**: Areas with low activation (weak filter response)
        
        This visualization is similar to techniques like Grad-CAM used in model interpretation, 
        which helps understand what features a CNN is using to make decisions.
        
        Notice how different filters highlight different aspects of the image - edge detectors highlight 
        boundaries, while blur filters respond more to areas of consistent intensity.
        """)
    
    # Tab 5: Receptive Field (now tab 5 since we removed Interactive Filter Creator)
    with tabs[4]:
        st.header("Receptive Field Visualization")
        st.write("""
        The receptive field is the region in the input image that affects a particular neuron in a layer.
        As we go deeper in the network, the receptive field grows, allowing deeper layers to "see" more
        of the input and recognize more complex patterns.
        """)
        
        # Network configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Image size
            image_size = st.slider(
                "Input Image Size:",
                min_value=32,
                max_value=256,
                value=64,
                step=32,
                key="rf_img_size"
            )
            
            # Number of layers
            num_layers = st.slider(
                "Number of Conv Layers:",
                min_value=1,
                max_value=5,
                value=3,
                key="rf_layers"
            )
        
        with col2:
            # Filter size
            filter_size = st.slider(
                "Filter Size:",
                min_value=3,
                max_value=7,
                value=3,
                step=2,  # only odd values
                key="rf_filter_size"
            )
            
            # Pooling size
            pooling_size = st.slider(
                "Pooling Size:",
                min_value=2,
                max_value=4,
                value=2,
                key="rf_pool_size"
            )
        
        # Calculate receptive fields
        layer_configs = visualize_receptive_field(
            image_size=image_size,
            layers=num_layers,
            filter_size=filter_size,
            pooling_size=pooling_size
        )
        
        # Display table of layer configurations
        st.subheader("Network Architecture")
        
        # Create a DataFrame for better display
        import pandas as pd
        
        df = pd.DataFrame(layer_configs)
        st.table(df)
        
        # Visualization of receptive field growth
        st.subheader("Receptive Field Growth Visualization")
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw the input image
        input_size = layer_configs[0]['size']
        from matplotlib import patches
        rect = patches.Rectangle((0, 0), input_size, input_size, fill=True, color='lightgray', alpha=0.5)
        ax.add_patch(rect)
        
        # Colors for different layers
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_configs)))
        
        # Draw receptive fields for each layer
        for i, layer in enumerate(layer_configs):
            rf_size = layer['rf_size']
            
            # Center the receptive field in the input image
            x = (input_size - rf_size) / 2
            y = (input_size - rf_size) / 2
            
            # Draw the receptive field
            rf_rect = patches.Rectangle(
                (x, y), rf_size, rf_size, 
                fill=True, 
                color=colors[i], 
                alpha=0.5,
                label=f"{layer['name']} (RF: {rf_size}×{rf_size})"
            )
            ax.add_patch(rf_rect)
        
        # Set limits
        ax.set_xlim(0, input_size)
        ax.set_ylim(0, input_size)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Input Width')
        ax.set_ylabel('Input Height')
        ax.set_title('Receptive Field Growth Across Layers')
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Set aspect ratio to be equal
        ax.set_aspect('equal')
        
        # Display the plot
        st.pyplot(fig)
        
        # Add explanation
        st.write("""
        ### Understanding Receptive Fields
        
        The receptive field is a crucial concept in CNNs:
        
        1. **Definition**: The receptive field of a neuron is the region in the input image that can
           influence that neuron's activation.
        
        2. **Growth**: As we go deeper in the network, the receptive field grows:
           - First layer neurons only see a small patch (filter size)
           - Each subsequent layer combines information from multiple neurons in the previous layer
           - Pooling layers increase the effective receptive field size
        
        3. **Importance**: The growth of receptive fields allows:
           - Early layers to detect simple features (edges, textures)
           - Middle layers to detect parts and patterns
           - Deep layers to recognize complex objects and scenes
        
        In the visualization:
        - The gray square represents the full input image
        - Each colored square shows how much of the input affects a neuron in that layer
        - Notice how the receptive field grows as we go deeper in the network
        
        This growth is why CNNs are so effective at hierarchical feature learning - they naturally
        build complexity by combining simpler features detected in earlier layers.
        """)