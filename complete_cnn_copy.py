import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import base64
from PIL import Image, ImageOps, ImageDraw, ImageFont
import io
import urllib.request
from io import BytesIO
import random
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
import filters

# Set matplotlib warning threshold higher to avoid excessive warnings
plt.rcParams['figure.max_open_warning'] = 50

# Helper function to display and close matplotlib figures
def safe_pyplot(fig):
    """Display a matplotlib figure in Streamlit and properly close it to free memory"""
    st.pyplot(fig)
    plt.close(fig)


# Sample images - We'll use embedded patterns for reliability
SAMPLE_IMAGES = []

# Generate a circle pattern
def generate_circle_pattern(size=64):
    image = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, size-10, size-10), fill=0)
    return {
        "name": "Circle Pattern",
        "image": image
    }

# Generate a grid pattern
def generate_grid_pattern(size=64, line_spacing=8):
    image = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(image)
    
    # Draw horizontal lines
    for y in range(0, size, line_spacing):
        draw.line([(0, y), (size, y)], fill=0, width=1)
    
    # Draw vertical lines
    for x in range(0, size, line_spacing):
        draw.line([(x, 0), (x, size)], fill=0, width=1)
    
    return {
        "name": "Grid Pattern",
        "image": image
    }

# Generate a triangle pattern
def generate_triangle_pattern(size=64):
    image = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(image)
    
    # Draw a triangle
    points = [(size//2, 10), (10, size-10), (size-10, size-10)]
    draw.polygon(points, fill=0)
    
    return {
        "name": "Triangle Pattern",
        "image": image
    }

# Generate a checkerboard pattern
def generate_checkerboard_pattern(size=64, square_size=8):
    image = Image.new('L', (size, size), color=255)
    pixels = image.load()
    
    # Fill the checkerboard
    for i in range(size):
        for j in range(size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                pixels[i, j] = 0
    
    return {
        "name": "Checkerboard Pattern",
        "image": image
    }

# Generate an X pattern
def generate_x_pattern(size=64):
    image = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(image)
    
    # Draw an X
    draw.line([(10, 10), (size-10, size-10)], fill=0, width=2)
    draw.line([(size-10, 10), (10, size-10)], fill=0, width=2)
    
    return {
        "name": "X Pattern",
        "image": image
    }

# Generate noise pattern
def generate_noise_pattern(size=64):
    # Create a random noise array
    noise = np.random.randint(0, 256, size=(size, size), dtype=np.uint8)
    image = Image.fromarray(noise)
    
    return {
        "name": "Noise Pattern",
        "image": image
    }

# Generate a simple text pattern
def generate_text_pattern(size=64, text="CNN"):
    image = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(image)
    
    # Add text
    try:
        # Try to load a font
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((size//4, size//3), text, fill=0, font=font)
    except:
        # If font loading fails, draw a simple shape instead
        draw.rectangle((size//4, size//3, 3*size//4, 2*size//3), outline=0)
    
    return {
        "name": "Text Pattern",
        "image": image
    }

# Generate a horizontal gradient
def generate_gradient_pattern(size=64):
    # Create a gradient from black to white
    gradient = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        gradient[:, i] = i * 255 // size
    
    image = Image.fromarray(gradient)
    
    return {
        "name": "Gradient Pattern",
        "image": image
    }

# Load sample images
try:
    from PIL import ImageDraw
    
    # Add generated patterns to sample images
    SAMPLE_IMAGES.append(generate_circle_pattern())
    SAMPLE_IMAGES.append(generate_grid_pattern())
    SAMPLE_IMAGES.append(generate_triangle_pattern())
    SAMPLE_IMAGES.append(generate_checkerboard_pattern())
    SAMPLE_IMAGES.append(generate_x_pattern())
    SAMPLE_IMAGES.append(generate_noise_pattern())
    SAMPLE_IMAGES.append(generate_text_pattern())
    SAMPLE_IMAGES.append(generate_gradient_pattern())
    
except Exception as e:
    st.error(f"Error generating sample images: {str(e)}")
    # Add a fallback sample image
    fallback = Image.new('L', (64, 64), 128)
    SAMPLE_IMAGES.append({"name": "Fallback Pattern", "image": fallback})


# Initialize with defaults to avoid unbound variable errors
filters_to_show = ["Edge Detection (Horizontal)", "Edge Detection (Vertical)", "Sobel (Horizontal)", "Sobel (Vertical)"]

# Activation function selection
activation_functions = {
    "ReLU": lambda x: np.maximum(0, x),
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "Tanh": lambda x: np.tanh(x),
    "Leaky ReLU": lambda x: np.maximum(0.1 * x, x)
}

# Pooling function selection
pooling_functions = {
    "Max Pooling": lambda x, pool_size: max_pooling(x, pool_size),
    "Average Pooling": lambda x, pool_size: avg_pooling(x, pool_size)
}

# Initialize with defaults to avoid unbound variable errors
selected_activation = "ReLU"
selected_pooling = "Max Pooling"
pooling_size = 2

all_filters = list(filters.COMMON_FILTERS.keys())


# class appmode:
#     def __init__(self, app_mode):
#         self.app_mode = app_mode
    
#     def call_app(self):
#         if self.app_mode == "Process Sample Images" or self.app_mode == "Upload Your Own Image":
#             filters_to_show = st.sidebar.multiselect(
#                 "Select filters to apply",
#                 all_filters,
#                 default=["Edge Detection (Horizontal)", "Edge Detection (Vertical)", "Sobel (Horizontal)", "Sobel (Vertical)"]
#             )
            
#             # Add "All Filters" option
#             if st.sidebar.checkbox("Use All Filters"):
#                 filters_to_show = all_filters
            
#             # Add custom filter option
#             custom_filter = st.sidebar.checkbox("Add Custom Filter")
            
#             if custom_filter:
#                 st.sidebar.markdown("Custom Filter Values (separated by commas)")
#                 custom_filter_rows = []
#                 for i in range(3):
#                     row_values = st.sidebar.text_input(f"Row {i+1}", value="0, 0, 0", key=f"custom_row_{i}")
#                     try:
#                         row = [float(x.strip()) for x in row_values.split(',')]
#                         if len(row) != 3:
#                             st.sidebar.warning(f"Row {i+1} must have 3 values")
#                             row = [0, 0, 0]
#                         custom_filter_rows.append(row)
#                     except:
#                         st.sidebar.warning(f"Invalid values in row {i+1}")
#                         custom_filter_rows.append([0, 0, 0])
                
#                 custom_filter_array = np.array(custom_filter_rows)
#                 filters.COMMON_FILTERS["Custom Filter"] = custom_filter_array
                
#                 if "Custom Filter" not in filters_to_show:
#                     filters_to_show.append("Custom Filter")



#         if self.app_mode == "Process Sample Images" or self.app_mode == "Upload Your Own Image":
#             st.sidebar.title("Activation & Pooling")
#             selected_activation = st.sidebar.selectbox(
#                 "Select activation function",
#                 list(activation_functions.keys())
#             )
            
#             selected_pooling = st.sidebar.selectbox(
#                 "Select pooling operation",
#                 list(pooling_functions.keys())
#             )
            
#             pooling_size = st.sidebar.slider("Pooling Size", 2, 5, 2, 1)

# This function is no longer needed since we're generating images directly
# It's kept as a stub for backward compatibility
def download_sample_image(url):
    """Legacy function - no longer used"""
    pass

def preprocess_image(image):
    """Preprocess image to grayscale"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    return img_array

def apply_convolution(image, kernel):
    """Apply convolution with a given kernel to the image"""
    # Apply convolution
    return signal.convolve2d(image, kernel, mode='valid', boundary='symm')

def animate_convolution(image, kernel):
    """Animate the convolution process step by step, showing the sliding window"""
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    
    # For an 8x8 image and 3x3 kernel, we should have a 6x6 output
    o_h = i_h - k_h + 1
    o_w = i_w - k_w + 1
    
    # Prepare output feature map
    output = np.zeros((o_h, o_w))
    
    # Prepare animation frames
    frames = []
    
    # Create a figure with three subplots side by side
    fig = plt.figure(figsize=(12, 5))
    
    # We want to show all steps for an 8x8 image
    # Process each position systematically (left-to-right, top-to-bottom)
    for i in range(o_h):
        for j in range(o_w):
            # Extract the current region
            region = image[i:i+k_h, j:j+k_w]
            
            # Calculate the result for this position
            output[i, j] = np.sum(region * kernel)
            
            # Create plot for this frame
            plt.clf()  # Clear figure
            
            # Left subplot: image with highlighted region
            ax1 = plt.subplot(1, 3, 1)
            ax1.imshow(image, cmap='gray')
            ax1.set_title('Input Image with Filter Window')
            ax1.axis('off')
            
            # Add grid lines to show pixel boundaries
            for ii in range(i_h+1):
                ax1.axhline(ii-0.5, color='white', linewidth=0.5, alpha=0.5)
            for jj in range(i_w+1):
                ax1.axvline(jj-0.5, color='white', linewidth=0.5, alpha=0.5)
            
            # Highlight the current region
            rect = patches.Rectangle((j-0.5, i-0.5), k_w, k_h, 
                               fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            
            # Middle subplot: kernel values and multiplication with region
            ax2 = plt.subplot(1, 3, 2)
            
            # Create a visual representation of the element-wise multiplication
            mult_result = region * kernel
            product_display = np.zeros((k_h, k_w*3))
            
            # Fill the display with region, kernel, and product
            for ii in range(k_h):
                for jj in range(k_w):
                    # First column: region value
                    product_display[ii, jj] = region[ii, jj]
                    # Second column: kernel value
                    product_display[ii, jj+k_w] = kernel[ii, jj]
                    # Third column: product
                    product_display[ii, jj+2*k_w] = mult_result[ii, jj]
            
            ax2.imshow(product_display, cmap='viridis')
            ax2.set_title('Region × Kernel = Product')
            
            # Add labels and text with actual values
            for ii in range(k_h):
                for jj in range(k_w):
                    # Region value
                    ax2.text(jj, ii, f"{region[ii, jj]:.2f}",
                             ha="center", va="center", color="w" if region[ii, jj] < 0.5 else "k")
                    # Kernel value
                    ax2.text(jj+k_w, ii, f"{kernel[ii, jj]:.2f}",
                             ha="center", va="center", color="w" if abs(kernel[ii, jj]) > 0.5 else "k")
                    # Product value
                    ax2.text(jj+2*k_w, ii, f"{mult_result[ii, jj]:.2f}",
                             ha="center", va="center", color="w" if abs(mult_result[ii, jj]) > 0.5 else "k")
            
            # Add column labels
            ax2.set_xticks([k_w/2-0.5, k_w*3/2-0.5, k_w*5/2-0.5])
            ax2.set_xticklabels(['Region', 'Kernel', 'Product'])
            ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax2.set_yticks([])
            
            # Right subplot: output feature map so far
            ax3 = plt.subplot(1, 3, 3)
            # Use a fixed color scale for consistent background colors
            vmin = -3  # Min value for color scale
            vmax = 3   # Max value for color scale
            im = ax3.imshow(output, cmap='inferno', vmin=vmin, vmax=vmax)
            ax3.set_title('Output Feature Map')
            
            # Add grid lines to show pixel boundaries
            for ii in range(o_h+1):
                ax3.axhline(ii-0.5, color='white', linewidth=0.5, alpha=0.5)
            for jj in range(o_w+1):
                ax3.axvline(jj-0.5, color='white', linewidth=0.5, alpha=0.5)
            
            # Highlight the current output position
            ax3.plot(j, i, 'o', color='red', markersize=8)
            
            # Add text showing the calculation result
            sum_value = np.sum(mult_result)
            calculation_text = f"Position ({i},{j}): Sum of products = {sum_value:.2f}"
            plt.figtext(0.5, 0.01, calculation_text, ha="center", fontsize=12)
            
            plt.tight_layout()
            
            # Capture the frame
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)  # Lower DPI for better performance
            buf.seek(0)
            frames.append(Image.open(buf))
    
    # Close the figure to free memory
    plt.close(fig)
    
    return frames, output

def apply_activation(feature_map, activation_name):
    """Apply activation function to feature map"""
    return activation_functions[activation_name](feature_map)

def max_pooling(feature_map, pool_size=2):
    """Apply max pooling operation"""
    h, w = feature_map.shape
    h_out = h // pool_size
    w_out = w // pool_size
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            output[i, j] = np.max(feature_map[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    
    return output

def avg_pooling(feature_map, pool_size=2):
    """Apply average pooling operation"""
    h, w = feature_map.shape
    h_out = h // pool_size
    w_out = w // pool_size
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            output[i, j] = np.mean(feature_map[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    
    return output

def apply_pooling(feature_map, pooling_name, pool_size=2):
    """Apply selected pooling operation"""
    return pooling_functions[pooling_name](feature_map, pool_size)

def visualize_complete_cnn_process(image, filters_dict, activation_name, pooling_name, pool_size=2, title="Complete CNN Process"):
    """Visualize the complete CNN process: convolution, activation, pooling, flattening, and fully connected"""
    
    # Apply all selected filters
    convolution_results = {}
    activation_results = {}
    pooling_results = {}
    
    for name, kernel in filters_dict.items():
        # Convolution
        conv_result = apply_convolution(image, kernel)
        convolution_results[name] = conv_result
        
        # Activation
        act_result = apply_activation(conv_result, activation_name)
        activation_results[name] = act_result
        
        # Pooling
        pool_result = apply_pooling(act_result, pooling_name, pool_size)
        pooling_results[name] = pool_result
    
    # Display original image first
    st.subheader("Original Image")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title("Original Image")
    ax.axis('off')
    safe_pyplot(fig)
    
    # Display the complete CNN visualization
    st.subheader(title)
    
    # Process each filter individually
    for name, kernel in filters_dict.items():
        st.markdown(f"### Processing with {name} Filter")
        
        # Set up the visualization with 4 columns for each stage
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Display the kernel
        with col1:
            st.markdown("**Filter Kernel**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(kernel, cmap='viridis')
            plt.colorbar(im, ax=ax)
            
            # Add kernel values as text
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    text = ax.text(j, i, f"{kernel[i, j]:.2f}",
                           ha="center", va="center", color="w" if abs(kernel[i, j]) > 0.5 else "k")
            
            ax.set_title(f"{name} Kernel")
            safe_pyplot(fig)
        
        # Column 2: Display convolution result
        with col2:
            st.markdown("**After Convolution**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(convolution_results[name], cmap='inferno')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Convolution Result")
            ax.axis('off')
            safe_pyplot(fig)
            
            # Add animation button for convolution process
            animate_button = st.button(f"▶️ Animate Convolution Process for {name}", key=f"animate_{name}")
            
            # If button is clicked, show the animation right here
            if animate_button:
                st.markdown("### Convolution Animation")
                st.markdown("Watch how the filter slides over the image to create the feature map")
                
                # Always use a fixed 8x8 grid for animation (for consistent demonstration)
                target_size = 8  # Small fixed size for clear visualization of convolution
                
                # Resize to exactly 8x8 for animation
                if image.shape != (target_size, target_size):
                    # Convert to PIL, resize to 8x8, then back to numpy
                    pil_img = Image.fromarray((image * 255).astype(np.uint8))
                    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.NEAREST)
                    small_image = np.array(pil_img) / 255.0
                else:
                    small_image = image.copy()
                
                # Make sure the kernel is 3x3 for animation (easier to visualize)
                anim_kernel = kernel.copy()
                if kernel.shape[0] > 3 or kernel.shape[1] > 3:
                    # Use a smaller version of the kernel (center part)
                    k_center_h = kernel.shape[0] // 2
                    k_center_w = kernel.shape[1] // 2
                    anim_kernel = kernel[k_center_h-1:k_center_h+2, k_center_w-1:k_center_w+2]
                
                # Run animation
                with st.spinner("Generating animation..."):
                    frames, _ = animate_convolution(small_image, anim_kernel)
                
                # Display animation
                if len(frames) > 0:
                    # Save as GIF with optimizations for speed but slower frame rate
                    gif_path = "convolution_animation.gif"
                    
                    # Reduce image size for faster loading
                    small_frames = []
                    for frame in frames:
                        small_frame = frame.resize((frame.width//2, frame.height//2), Image.Resampling.NEAREST)
                        small_frames.append(small_frame)
                    
                    small_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=small_frames[1:],
                        optimize=True,
                        duration=500,  # milliseconds per frame (slower animation)
                        loop=1  # Run only once (1) instead of infinitely (0)
                    )
                    
                    # Use an expander to contain the animation with a collapse option
                    with st.expander("**Convolution Process Animation** (click to expand/collapse)", expanded=True):
                        # Add explanation
                        st.markdown("""
                        This animation shows how the filter kernel slides over the input image to produce the output feature map.
                        Each frame displays:
                        - Left: Input image with the current filter position highlighted in red
                        - Middle: The filter kernel values
                        - Right: The resulting feature map with the current position highlighted
                        """)
                        
                        # Display the animated GIF
                        with open(gif_path, "rb") as file:
                            st.image(file.read(), caption="Click outside this area to continue browsing", use_container_width=True)
        
        # Column 3: Display activation result
        with col3:
            st.markdown(f"**After {activation_name}**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(activation_results[name], cmap='plasma')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{activation_name} Result")
            ax.axis('off')
            safe_pyplot(fig)
            
        # Column 4: Display pooling result
        with col4:
            st.markdown(f"**After {pooling_name}**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(pooling_results[name], cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{pooling_name} Result")
            ax.axis('off')
            safe_pyplot(fig)
        
        # Display 3D visualizations for each stage
        st.markdown("**3D Visualizations of Feature Maps**")
        st.markdown("""
        The 3D visualizations below show the activation values in each feature map as a surface, where:
        - **Height (Z-axis)** represents the activation strength at each position
        - **Colors** indicate activation intensity (brighter = stronger activation)
        - **Contour lines** show areas of equal activation values
        - **Peaks** reveal where features were strongly detected
        - **Valleys** show areas with weak or negative responses

        You can rotate, zoom, and pan these 3D plots to explore how the filter responds to different parts of the image.
        """)
        col1, col2, col3 = st.columns(3)
        
        # 3D visualization of convolution result with Plotly (interactive)
        with col1:
            st.markdown("**Convolution Feature Map (Interactive)**")
            
            # Create interactive 3D surface plot with Plotly
            x = np.arange(0, convolution_results[name].shape[1])
            y = np.arange(0, convolution_results[name].shape[0])
            x_mesh, y_mesh = np.meshgrid(x, y)
            
            # Create interactive 3D surface
            fig = go.Figure(data=[go.Surface(
                z=convolution_results[name],
                x=x_mesh, 
                y=y_mesh,
                colorscale='Inferno',
                colorbar=dict(title="Activation"),
                contours = {
                    "z": {"show": True, "start": 0, "end": 1, "size": 0.05}
                }
            )])
            
            fig.update_layout(
                title=f"3D View of Convolution",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Activation',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1))
                ),
                width=500,
                height=500,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig)
        
        # 3D visualization of activation result with Plotly (interactive)
        with col2:
            st.markdown(f"**{activation_name} Feature Map (Interactive)**")
            
            # Create interactive 3D surface plot with Plotly
            x = np.arange(0, activation_results[name].shape[1])
            y = np.arange(0, activation_results[name].shape[0])
            x_mesh, y_mesh = np.meshgrid(x, y)
            
            # Create interactive 3D surface
            fig = go.Figure(data=[go.Surface(
                z=activation_results[name],
                x=x_mesh, 
                y=y_mesh,
                colorscale='Plasma',
                colorbar=dict(title="Activation"),
                contours = {
                    "z": {"show": True, "start": 0, "end": 1, "size": 0.05}
                }
            )])
            
            fig.update_layout(
                title=f"3D View of {activation_name}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Activation',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1))
                ),
                width=500,
                height=500,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig)
        
        # 3D visualization of pooling result with Plotly (interactive)
        with col3:
            st.markdown(f"**{pooling_name} Feature Map (Interactive)**")
            
            # Create interactive 3D surface plot with Plotly
            x = np.arange(0, pooling_results[name].shape[1])
            y = np.arange(0, pooling_results[name].shape[0])
            x_mesh, y_mesh = np.meshgrid(x, y)
            
            # Create interactive 3D surface
            fig = go.Figure(data=[go.Surface(
                z=pooling_results[name],
                x=x_mesh, 
                y=y_mesh,
                colorscale='Viridis',
                colorbar=dict(title="Activation"),
                contours = {
                    "z": {"show": True, "start": 0, "end": 1, "size": 0.05}
                }
            )])
            
            fig.update_layout(
                title=f"3D View of {pooling_name}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Activation',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1))
                ),
                width=500,
                height=500,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig)
        
        # Flattening and Fully Connected Visualization
        st.markdown("### Flattening and Fully Connected Layer")
        
        # Flatten the pooling output
        flattened = pooling_results[name].flatten()
        
        # Display flattened data
        st.markdown(f"**Flattened Output (showing first 40 values of {len(flattened)} total)**")
        
        # Create a bar chart of flattened values
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(min(40, len(flattened))), flattened[:40])
        ax.set_title(f"Flattened Values (first 40)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Simulate a fully connected layer
        num_output_nodes = 10  # Assuming 10 classes (like MNIST)
        
        # Create random weights and biases for demonstration
        np.random.seed(42)  # For reproducibility
        fc_weights = np.random.randn(len(flattened), num_output_nodes) * 0.1
        fc_biases = np.random.randn(num_output_nodes) * 0.05
        
        # Compute the output of the fully connected layer
        fc_output = np.dot(flattened, fc_weights) + fc_biases
        
        # Apply softmax to get probabilities
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        
        probabilities = softmax(fc_output)
        
        # Display the final output probabilities
        st.markdown("**Final Output (Class Probabilities)**")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(num_output_nodes), probabilities)
        ax.set_xticks(range(num_output_nodes))
        ax.set_xticklabels([f"Class {i}" for i in range(num_output_nodes)])
        ax.set_title("Class Probabilities")
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Compare all pooling results side by side
    st.subheader("Side-by-Side Comparison of All Feature Maps After Processing")
    
    # Determine grid layout based on number of filters
    n_filters = len(filters_dict)
    n_cols = min(3, max(1, n_filters))  # Ensure at least 1 column
    n_rows = 1 if n_filters == 0 else (n_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    for i, (name, result) in enumerate(pooling_results.items()):
        if i < len(axs):
            im = axs[i].imshow(result, cmap='viridis')
            plt.colorbar(im, ax=axs[i])
            axs[i].set_title(f"{name} (after {pooling_name})")
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # No animation section needed here, as animations now display directly after each button click
    
    # Feature activation heatmap
    st.subheader("Feature Activation Analysis")
    
    # Create a matrix of filter activations
    activation_values = []
    filter_names = []
    metrics = ["Avg Activation", "Max Activation", "Variance", "% Positive Values"]
    
    for name, result in pooling_results.items():
        # Calculate metrics
        avg_activation = np.mean(np.abs(result))
        max_activation = np.max(np.abs(result))
        var_activation = np.var(result)
        positive_pct = np.sum(result > 0) / result.size * 100
        
        activation_values.append([avg_activation, max_activation, var_activation, positive_pct])
        filter_names.append(name)
    
    # Create heatmap
    activation_values = np.array(activation_values)
    
    fig, ax = plt.subplots(figsize=(10, len(filter_names) * 0.5 + 2))
    im = ax.imshow(activation_values, cmap='YlOrRd')
    
    # Set labels
    ax.set_yticks(np.arange(len(filter_names)))
    ax.set_yticklabels(filter_names)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add values to cells
    for i in range(len(filter_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{activation_values[i, j]:.2f}",
                        ha="center", va="center", color="black")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Return the results
    return {
        "convolution": convolution_results,
        "activation": activation_results,
        "pooling": pooling_results
    }

def generate_custom_filter():
    """Generate a custom filter based on user inputs"""
    st.subheader("Custom Filter Builder")
    
    # Filter size
    filter_size = st.slider("Filter Size", 3, 7, 3, 2, key="custom_filter_size")
    
    # Filter type
    filter_type = st.selectbox("Filter Type", [
        "Manual Entry", 
        "Edge Detection", 
        "Blur", 
        "Sharpen", 
        "Emboss", 
        "Random"
    ])
    
    if filter_type == "Manual Entry":
        # Create a grid of input boxes for filter values
        st.markdown("Enter filter values:")
        
        filter_values = []
        for i in range(filter_size):
            row = []
            cols = st.columns(filter_size)
            for j in range(filter_size):
                with cols[j]:
                    val = st.number_input(f"({i},{j})", value=0.0, format="%.2f", key=f"filter_{i}_{j}")
                    row.append(val)
            filter_values.append(row)
        
        custom_filter = np.array(filter_values)
        
    elif filter_type == "Edge Detection":
        direction = st.selectbox("Direction", ["Horizontal", "Vertical", "All Directions"])
        
        if direction == "Horizontal":
            custom_filter = np.zeros((filter_size, filter_size))
            custom_filter[0, :] = -1
            custom_filter[filter_size-1, :] = 1
            
        elif direction == "Vertical":
            custom_filter = np.zeros((filter_size, filter_size))
            custom_filter[:, 0] = -1
            custom_filter[:, filter_size-1] = 1
            
        else:  # All Directions
            custom_filter = np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ])
            
            if filter_size > 3:
                # Pad with zeros for larger filters
                padded = np.zeros((filter_size, filter_size))
                start = (filter_size - 3) // 2
                padded[start:start+3, start:start+3] = custom_filter
                custom_filter = padded
    
    elif filter_type == "Blur":
        blur_type = st.selectbox("Blur Type", ["Box Blur", "Gaussian Blur"])
        
        if blur_type == "Box Blur":
            custom_filter = np.ones((filter_size, filter_size)) / (filter_size * filter_size)
            
        else:  # Gaussian Blur
            # Create a simple approximation of a Gaussian kernel
            sigma = st.slider("Sigma (Blur Strength)", 0.5, 2.0, 1.0, 0.1)
            x = np.linspace(-2, 2, filter_size)
            y = np.linspace(-2, 2, filter_size)
            x, y = np.meshgrid(x, y)
            d = np.sqrt(x*x + y*y)
            custom_filter = np.exp(-(d**2)/(2*sigma**2))
            custom_filter = custom_filter / custom_filter.sum()  # Normalize
    
    elif filter_type == "Sharpen":
        # Create a sharpening filter
        custom_filter = np.zeros((filter_size, filter_size))
        center = filter_size // 2
        
        # Set center value
        custom_filter[center, center] = filter_size + 0.5
        
        # Set negative values around center
        for i in range(filter_size):
            for j in range(filter_size):
                if i == center and j == center:
                    continue
                if abs(i - center) <= 1 and abs(j - center) <= 1:
                    custom_filter[i, j] = -1
    
    elif filter_type == "Emboss":
        direction = st.selectbox("Direction", ["Top-Left to Bottom-Right", "Bottom-Left to Top-Right"])
        
        custom_filter = np.zeros((filter_size, filter_size))
        
        if direction == "Top-Left to Bottom-Right":
            for i in range(filter_size):
                for j in range(filter_size):
                    if i < j:
                        custom_filter[i, j] = 1
                    elif i > j:
                        custom_filter[i, j] = -1
        else:
            for i in range(filter_size):
                for j in range(filter_size):
                    if i + j < filter_size - 1:
                        custom_filter[i, j] = 1
                    elif i + j > filter_size - 1:
                        custom_filter[i, j] = -1
    
    else:  # Random
        # Create a random filter
        custom_filter = np.random.randn(filter_size, filter_size)
        # Normalize to sum to 1
        if st.checkbox("Normalize Filter", value=True):
            custom_filter = custom_filter / np.sum(np.abs(custom_filter))
    
    # Display the filter
    st.subheader("Filter Preview")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(custom_filter, cmap='viridis')
    plt.colorbar(im, ax=ax)
    
    # Add filter values as text
    for i in range(filter_size):
        for j in range(filter_size):
            text = ax.text(j, i, f"{custom_filter[i, j]:.2f}",
                   ha="center", va="center", color="w" if abs(custom_filter[i, j]) > 0.5 else "k")
    
    ax.set_title("Custom Filter")
    st.pyplot(fig)
    
    # Option to normalize the filter
    if st.checkbox("Normalize Filter Values", value=False):
        norm_type = st.radio("Normalization Type", ["Sum to 1", "Max Value to 1", "Range [-1, 1]"])
        
        if norm_type == "Sum to 1":
            custom_filter = custom_filter / np.sum(np.abs(custom_filter))
        elif norm_type == "Max Value to 1":
            custom_filter = custom_filter / np.max(np.abs(custom_filter))
        else:  # Range [-1, 1]
            max_val = np.max(np.abs(custom_filter))
            if max_val > 0:
                custom_filter = custom_filter / max_val
        
        # Display the normalized filter
        st.markdown("**Normalized Filter**")
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(custom_filter, cmap='viridis')
        plt.colorbar(im, ax=ax)
        
        # Add filter values as text
        for i in range(filter_size):
            for j in range(filter_size):
                text = ax.text(j, i, f"{custom_filter[i, j]:.2f}",
                       ha="center", va="center", color="w" if abs(custom_filter[i, j]) > 0.5 else "k")
        
        ax.set_title("Normalized Custom Filter")
        st.pyplot(fig)
    
    # Test the filter with different activation functions
    st.subheader("Activation Function Impact")
    
    # Allow user to select a test image
    test_image_name = st.selectbox("Select a test image", [img["name"] for img in SAMPLE_IMAGES])
    test_image_data = next(img for img in SAMPLE_IMAGES if img["name"] == test_image_name)
    
    # Get image directly from our sample images
    test_image = test_image_data["image"]
    test_image_array = preprocess_image(test_image)
    
    # Apply the custom filter
    conv_result = apply_convolution(test_image_array, custom_filter)
    
    # Display original and convolution result
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(test_image, caption=test_image_name, use_container_width=True)
    
    with col2:
        st.markdown("**Convolution Result (Interactive)**")
        
        # Create interactive 3D surface plot with Plotly
        x = np.arange(0, conv_result.shape[1])
        y = np.arange(0, conv_result.shape[0])
        x_mesh, y_mesh = np.meshgrid(x, y)
        
        # Create interactive 3D surface
        fig = go.Figure(data=[go.Surface(
            z=conv_result,
            x=x_mesh, 
            y=y_mesh,
            colorscale='Inferno',
            colorbar=dict(title="Activation"),
            contours = {
                "z": {"show": True, "start": 0, "end": 1, "size": 0.05}
            }
        )])
        
        fig.update_layout(
            title="Interactive Feature Map",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Activation',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
            ),
            width=500,
            height=500,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig)
    
    # Apply different activation functions
    st.markdown("### Comparing Activation Functions")
    
    cols = st.columns(len(activation_functions))
    
    for i, (name, func) in enumerate(activation_functions.items()):
        with cols[i]:
            activation_result = func(conv_result)
            
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(activation_result, cmap='plasma')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{name} Result")
            ax.axis('off')
            st.pyplot(fig)
    
    # Apply different pooling operations
    st.markdown("### Comparing Pooling Operations")
    
    cols = st.columns(len(pooling_functions))
    
    # Use ReLU activated result for pooling comparison
    relu_result = activation_functions["ReLU"](conv_result)
    
    for i, (name, func) in enumerate(pooling_functions.items()):
        with cols[i]:
            pooling_result = func(relu_result, 2)  # Use pool size 2
            
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(pooling_result, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{name} Result")
            ax.axis('off')
            st.pyplot(fig)
    
    return custom_filter

def visualize_different_activation_functions(feature_map):
    """Visualize the same feature map with different activation functions"""
    st.subheader("Comparing Different Activation Functions")
    
    # Create columns for each activation function
    cols = st.columns(len(activation_functions))
    
    for i, (name, func) in enumerate(activation_functions.items()):
        with cols[i]:
            activation_result = func(feature_map)
            
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(activation_result, cmap='plasma')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{name} Result")
            ax.axis('off')
            st.pyplot(fig)
            
            # Display a histogram of activation values
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(activation_result.flatten(), bins=30)
            ax.set_title(f"{name} Value Distribution")
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

def visualize_different_pooling_operations(feature_map):
    """Visualize the same feature map with different pooling operations"""
    st.subheader("Comparing Different Pooling Operations")
    
    # Create columns for each pooling operation
    cols = st.columns(len(pooling_functions))
    
    for pool_size in [2, 3, 4]:
        st.markdown(f"#### Pool Size = {pool_size}")
        cols = st.columns(len(pooling_functions))
        
        for i, (name, func) in enumerate(pooling_functions.items()):
            with cols[i]:
                try:
                    pooling_result = func(feature_map, pool_size)
                    
                    st.markdown(f"**{name}**")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(pooling_result, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"{name} Result")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Show reduction in dimensionality
                    st.markdown(f"Dimensionality: {feature_map.shape} → {pooling_result.shape}")
                    st.markdown(f"Reduction: {feature_map.size/pooling_result.size:.1f}x")
                except:
                    st.error(f"Feature map too small for pool size {pool_size}")

class appmode:
    def __init__(self, app_mode):
        self.app_mode = app_mode
    
    def call_app(self):
        if self.app_mode == "Process Sample Images" or self.app_mode == "Upload Your Own Image":
            filters_to_show = st.sidebar.multiselect(
                "Select filters to apply",
                all_filters,
                default=["Edge Detection (Horizontal)", "Edge Detection (Vertical)", "Sobel (Horizontal)", "Sobel (Vertical)"]
            )
            
            # Add "All Filters" option
            if st.sidebar.checkbox("Use All Filters"):
                filters_to_show = all_filters
            
            # Add custom filter option
            custom_filter = st.sidebar.checkbox("Add Custom Filter")
            
            if custom_filter:
                st.sidebar.markdown("Custom Filter Values (separated by commas)")
                custom_filter_rows = []
                for i in range(3):
                    row_values = st.sidebar.text_input(f"Row {i+1}", value="0, 0, 0", key=f"custom_row_{i}")
                    try:
                        row = [float(x.strip()) for x in row_values.split(',')]
                        if len(row) != 3:
                            st.sidebar.warning(f"Row {i+1} must have 3 values")
                            row = [0, 0, 0]
                        custom_filter_rows.append(row)
                    except:
                        st.sidebar.warning(f"Invalid values in row {i+1}")
                        custom_filter_rows.append([0, 0, 0])
                
                custom_filter_array = np.array(custom_filter_rows)
                filters.COMMON_FILTERS["Custom Filter"] = custom_filter_array
                
                if "Custom Filter" not in filters_to_show:
                    filters_to_show.append("Custom Filter")



        if self.app_mode == "Process Sample Images" or self.app_mode == "Upload Your Own Image":
            st.sidebar.title("Activation & Pooling")
            selected_activation = st.sidebar.selectbox(
                "Select activation function",
                list(activation_functions.keys())
            )
            
            selected_pooling = st.sidebar.selectbox(
                "Select pooling operation",
                list(pooling_functions.keys())
            )
            
            pooling_size = st.sidebar.slider("Pooling Size", 2, 5, 2, 1)

        # Introduction section
        if self.app_mode == "Introduction":
            st.header("How CNNs Extract Features from Images")
            
            st.markdown("""
            ### Understanding the Complete CNN Pipeline
            
            Convolutional Neural Networks (CNNs) process images through a series of transformations:
            
            1. **Convolution**: Applies filters to detect features
            2. **Activation**: Introduces non-linearity (ReLU, Sigmoid, Tanh)
            3. **Pooling**: Reduces dimensions while preserving important information
            4. **Flattening**: Converts 2D feature maps to 1D vectors
            5. **Fully Connected Layers**: Make predictions based on extracted features
            
            This application demonstrates each step of this process on real images at their original resolution.
            You can select different filters, activation functions, and pooling methods to see how they affect the feature extraction.
            """)
            
            # Display CNN architecture diagram
            st.subheader("CNN Architecture")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Define components and their positions
            components = [
                {'name': 'Input\nImage', 'pos': 0, 'width': 0.5, 'height': 1, 'color': 'lightblue'},
                {'name': 'Convolution\nLayers', 'pos': 1.5, 'width': 0.5, 'height': 1, 'color': 'lightgreen'},
                {'name': 'Activation\n(ReLU, etc.)', 'pos': 3, 'width': 0.5, 'height': 1, 'color': 'pink'},
                {'name': 'Pooling\nLayers', 'pos': 4.5, 'width': 0.5, 'height': 1, 'color': 'lightyellow'},
                {'name': 'Flatten', 'pos': 6, 'width': 0.5, 'height': 1, 'color': 'lightgray'},
                {'name': 'Fully Connected\nLayers', 'pos': 7.5, 'width': 0.5, 'height': 1, 'color': 'lightcoral'},
                {'name': 'Output\n(Classes)', 'pos': 9, 'width': 0.5, 'height': 1, 'color': 'lightgreen'}
            ]
            
            # Plot components
            for comp in components:
                ax.add_patch(patches.Rectangle((comp['pos'], 0.5), comp['width'], comp['height'], 
                                        facecolor=comp['color'], edgecolor='black'))
                ax.text(comp['pos'] + comp['width']/2, 1, comp['name'], 
                    ha='center', va='center', fontsize=9)
            
            # Add arrows connecting components
            for i in range(len(components)-1):
                start_x = components[i]['pos'] + components[i]['width']
                end_x = components[i+1]['pos']
                ax.arrow(start_x, 1, end_x - start_x - 0.05, 0, 
                        head_width=0.1, head_length=0.05, fc='black', ec='black')
            
            # Set axis limits and turn off axis
            ax.set_xlim(-0.5, 10)
            ax.set_ylim(0, 2)
            ax.axis('off')
            
            st.pyplot(fig)
            
            # Show sample convolutional filters
            st.subheader("Common CNN Filters")
            
            # Create a grid of sample filters
            n_cols = 3
            n_rows = (len(filters.COMMON_FILTERS) + n_cols - 1) // n_cols
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axs = axs.flatten()
            
            for i, (name, kernel) in enumerate(filters.COMMON_FILTERS.items()):
                if i < len(axs):
                    im = axs[i].imshow(kernel, cmap='viridis')
                    plt.colorbar(im, ax=axs[i])
                    axs[i].set_title(name)
                    
                    # Add kernel values as text
                    for r in range(kernel.shape[0]):
                        for c in range(kernel.shape[1]):
                            text = axs[i].text(c, r, f"{kernel[r, c]:.2f}",
                                ha="center", va="center", color="w" if np.abs(kernel[r, c]) > 0.5 else "k")
            
            # Hide unused subplots
            for i in range(len(filters.COMMON_FILTERS), len(axs)):
                axs[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explain activation functions
            st.subheader("Activation Functions")
            
            # Generate a range of values for visualization
            x = np.linspace(-5, 5, 1000)
            
            # Plot each activation function
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            
            for i, (name, func) in enumerate(activation_functions.items()):
                y = func(x)
                axs[i].plot(x, y)
                axs[i].set_title(name)
                axs[i].grid(True)
                axs[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axs[i].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            ### Activation Functions:
            
            - **ReLU (Rectified Linear Unit)**: Outputs the input directly if positive, otherwise outputs zero
            - **Sigmoid**: Squashes values between 0 and 1, useful for binary classification
            - **Tanh**: Squashes values between -1 and 1, centered at zero
            - **Leaky ReLU**: Similar to ReLU but allows small negative values to pass through
            
            Activation functions add non-linearity to the network, allowing it to learn complex patterns.
            """)
            
            # Explain pooling operations
            st.subheader("Pooling Operations")
            
            # Create a sample feature map
            sample_map = np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ])
            
            # Apply max and average pooling
            max_pool = max_pooling(sample_map, 2)
            avg_pool = avg_pooling(sample_map, 2)
            
            # Display original and pooled maps
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original map
            im1 = ax1.imshow(sample_map, cmap='viridis')
            ax1.set_title("Original Feature Map")
            for i in range(4):
                for j in range(4):
                    ax1.text(j, i, f"{sample_map[i, j]}", ha="center", va="center", color="w")
            
            # Draw pooling regions
            for i in range(0, 4, 2):
                for j in range(0, 4, 2):
                    rect = patches.Rectangle((j-0.5, i-0.5), 2, 2, fill=False, edgecolor='r', linewidth=2)
                    ax1.add_patch(rect)
            
            # Max pooling
            im2 = ax2.imshow(max_pool, cmap='viridis')
            ax2.set_title("Max Pooling (2x2)")
            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, f"{max_pool[i, j]}", ha="center", va="center", color="w")
            
            # Average pooling
            im3 = ax3.imshow(avg_pool, cmap='viridis')
            ax3.set_title("Average Pooling (2x2)")
            for i in range(2):
                for j in range(2):
                    ax3.text(j, i, f"{avg_pool[i, j]:.1f}", ha="center", va="center", color="w")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            ### Pooling Operations:
            
            - **Max Pooling**: Takes the maximum value from each pooling region
            - **Average Pooling**: Takes the average of all values in each pooling region
            
            Pooling reduces the spatial dimensions of the feature maps, which:
            - Reduces the number of parameters and computation
            - Provides some translation invariance
            - Helps prevent overfitting
            
            The pool size determines how much downsampling occurs.
            """)
            
            # Explain flattening and fully connected layers
            st.subheader("Flattening and Fully Connected Layers")
            
            st.markdown("""
            After convolution, activation, and pooling, the resulting feature maps are **flattened** into a 1D vector.
            This vector is then passed through one or more **fully connected layers** to make the final prediction.
            
            For example, if we have 16 feature maps of size 4x4 after pooling, flattening would produce a vector of length 256 (16 × 4 × 4).
            The fully connected layer applies weights to this vector to produce the final class scores or probabilities.
            
            In this application, we simulate this process to show how the extracted features would be used for classification.
            """)
            
            # Complete CNN processing animation
            st.subheader("The Complete CNN Process")
            
            st.markdown("""
            Explore the different sections of this application to see how these operations work together on real images.
            You can:
            
            - Process sample images with different filters and see the results at each stage
            - Upload your own images and see how CNNs would process them
            - Build custom filters to see how they extract different features
            
            This interactive visualization helps understand why CNNs are so effective for image classification and feature extraction.
            """)

        # Process Sample Images section
        elif self.app_mode == "Process Sample Images":
            st.header("Process Sample Images")
            
            st.markdown("""
            See how CNNs process sample images through each stage: convolution, activation, pooling, and fully connected layers.
            """)
            
            # Create image selection
            sample_images = [img["name"] for img in SAMPLE_IMAGES]
            selected_image_name = st.selectbox("Select a sample image", sample_images)
            
            # Find the selected image data
            selected_image_data = next(img for img in SAMPLE_IMAGES if img["name"] == selected_image_name)
            
            # Get the image directly from our sample images
            image = selected_image_data["image"]
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption=selected_image_name, use_container_width=True)
            
            # Process image (convert to grayscale but maintain original size)
            processed_img = preprocess_image(image)
            
            # Create a dictionary of selected filters
            selected_filters = {name: filters.COMMON_FILTERS[name] for name in filters_to_show}
            
            # Visualize complete CNN process
            results = visualize_complete_cnn_process(
                processed_img, 
                selected_filters, 
                selected_activation, 
                selected_pooling, 
                pooling_size, 
                title=f"Complete CNN Processing for {selected_image_name}"
            )
            
            # Display activation function comparison for one filter
            st.header("Activation Function Comparison")
            
            # Select a filter to compare activation functions
            activation_filter_name = st.selectbox("Select a filter to compare activation functions", filters_to_show)
            
            # Get the convolution result for the selected filter
            convolution_result = results["convolution"][activation_filter_name]
            
            # Visualize different activation functions
            visualize_different_activation_functions(convolution_result)
            
            # Display pooling operation comparison
            st.header("Pooling Operation Comparison")
            
            # Get the ReLU result for the selected filter
            relu_result = results["activation"][activation_filter_name]
            
            # Visualize different pooling operations
            visualize_different_pooling_operations(relu_result)
            
            # Multiple Layer Visualization
            st.header("Multiple Layer Processing")
            
            st.markdown("""
            In real CNNs, multiple convolutional layers are stacked to extract increasingly complex features.
            Here's a simulation of what happens when we stack two convolutional layers:
            """)
            
            # Select two filters to demonstrate stacked convolutions
            first_layer_filter = "Edge Detection (Horizontal)"
            second_layer_filters = ["Edge Detection (Vertical)", "Sobel (Horizontal)"]
            
            # First layer of convolution
            first_layer_result = apply_convolution(processed_img, filters.COMMON_FILTERS[first_layer_filter])
            first_layer_activation = apply_activation(first_layer_result, selected_activation)
            first_layer_pooling = apply_pooling(first_layer_activation, selected_pooling, pooling_size)
            
            # Display first layer results
            st.subheader("First Convolutional Layer")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**{first_layer_filter} Filter**")
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(filters.COMMON_FILTERS[first_layer_filter], cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title("Filter")
                st.pyplot(fig)
            
            with col2:
                st.markdown("**After First Convolution**")
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(first_layer_result, cmap='inferno')
                plt.colorbar(im, ax=ax)
                ax.set_title("First Layer Feature Map")
                ax.axis('off')
                st.pyplot(fig)
            
            with col3:
                st.markdown(f"**After {selected_activation} & {selected_pooling}**")
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(first_layer_pooling, cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title("First Layer Output")
                ax.axis('off')
                st.pyplot(fig)
            
            # Apply second layer of convolution to the first layer output
            st.subheader("Second Convolutional Layer")
            
            # Process with each second layer filter
            cols = st.columns(len(second_layer_filters))
            
            for i, filter_name in enumerate(second_layer_filters):
                with cols[i]:
                    # Apply second layer of convolution
                    second_layer_filter = filters.COMMON_FILTERS[filter_name]
                    second_layer_result = apply_convolution(first_layer_pooling, second_layer_filter)
                    second_layer_activation = apply_activation(second_layer_result, selected_activation)
                    
                    # Display filter
                    st.markdown(f"**{filter_name} Filter**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(second_layer_filter, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.set_title("Second Layer Filter")
                    st.pyplot(fig)
                    
                    # Display result
                    st.markdown("**Second Layer Feature Map**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(second_layer_activation, cmap='plasma')
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"Output of {filter_name}")
                    ax.axis('off')
                    st.pyplot(fig)
            
            st.markdown("""
            Notice how the second layer detects more complex patterns by building upon the features extracted by the first layer.
            This hierarchical feature extraction is what makes CNNs so powerful for image analysis.
            """)

        # Upload Your Own Image section
        elif self.app_mode == "Upload Your Own Image":
            st.header("Upload Your Own Image")
            
            st.markdown("""
            Upload your own image to see how CNNs process it through all stages:
            - Convolution with various filters
            - Activation with different functions
            - Pooling operations
            - Flattening and fully connected layers
            
            The image will be processed at its original resolution (or slightly reduced for very large images).
            """)
            
            # Upload image
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])
            
            if uploaded_file is not None:
                # Display original image
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Preprocess image
                try:
                    # If image is very large, resize to a more manageable size while preserving aspect ratio
                    w, h = image.size
                    max_size = 400
                    if w > max_size or h > max_size:
                        ratio = min(max_size / w, max_size / h)
                        new_size = (int(w * ratio), int(h * ratio))
                        image = image.resize(new_size, Image.Resampling.BICUBIC)
                        st.info(f"Image was resized from {w}x{h} to {new_size[0]}x{new_size[1]} for better performance.")
                    
                    processed_img = preprocess_image(image)
                    
                    # Create a dictionary of selected filters
                    selected_filters = {name: filters.COMMON_FILTERS[name] for name in filters_to_show}
                    
                    # Visualize complete CNN process
                    results = visualize_complete_cnn_process(
                        processed_img, 
                        selected_filters, 
                        selected_activation, 
                        selected_pooling, 
                        pooling_size, 
                        title="Complete CNN Processing for Your Image"
                    )
                    
                    # Display activation function comparison for one filter
                    st.header("Activation Function Comparison")
                    
                    # Select a filter to compare activation functions
                    activation_filter_name = st.selectbox("Select a filter to compare activation functions", filters_to_show)
                    
                    # Get the convolution result for the selected filter
                    convolution_result = results["convolution"][activation_filter_name]
                    
                    # Visualize different activation functions
                    visualize_different_activation_functions(convolution_result)
                    
                    # Display pooling operation comparison
                    st.header("Pooling Operation Comparison")
                    
                    # Get the ReLU result for the selected filter
                    relu_result = results["activation"][activation_filter_name]
                    
                    # Visualize different pooling operations
                    visualize_different_pooling_operations(relu_result)
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.info("Please try uploading a different image.")

        # Custom Filter Builder section
        elif self.app_mode == "Custom Filter Builder":
            st.header("Custom Filter Builder")
            
            st.markdown("""
            Create your own convolutional filter and test it on sample images.
            This tool helps you understand how different filter designs extract different features.
            """)
            
            # Generate and display the custom filter
            custom_filter = generate_custom_filter()
            
        def main():
            # Main app layout
            st.title("CNN Visualization Explorer")
            
            # Top-level app options
            app_mode = st.sidebar.selectbox(
                "What would you like to do?",
                ["About", "Process Sample Images", "Upload Your Own Image", "Custom Filter Builder"]
            )
            
            # About section
            if app_mode == "About":
                st.header("CNN Visualization Explorer")
                
                st.markdown("""
                This app demonstrates how Convolutional Neural Networks (CNNs) process images through different layers.
                
                ### What You Can Do:
                - Process sample images with different filters and see the results at each stage
                - Upload your own images and see how CNNs would process them
                - Build custom filters to see how they extract different features
                
                This interactive visualization helps understand why CNNs are so effective for image classification and feature extraction.
                """)

            # Process Sample Images section
            elif app_mode == "Process Sample Images":
                st.header("Process Sample Images")
                
                st.markdown("""
                See how CNNs process sample images through each stage: convolution, activation, pooling, and fully connected layers.
                """)
                
                # Create image selection
                sample_images = [img["name"] for img in SAMPLE_IMAGES]
                selected_image_name = st.selectbox("Select a sample image", sample_images)
                
                # Find the selected image data
                selected_image_data = next(img for img in SAMPLE_IMAGES if img["name"] == selected_image_name)
                
                # Get the image directly from our sample images
                image = selected_image_data["image"]
                
                # Display original image
                st.subheader("Original Image")
                st.image(image, caption=selected_image_name, use_container_width=True)
                
                # Process image (convert to grayscale but maintain original size)
                processed_img = preprocess_image(image)
                
                # CNN Processing Options
                st.subheader("CNN Layer Settings")
                
                # Filter selection
                st.markdown("**Select filters to apply:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    filter1 = st.checkbox("Edge Detection (Horizontal)", value=True)
                    filter4 = st.checkbox("Gaussian Blur", value=False)
                
                with col2:
                    filter2 = st.checkbox("Edge Detection (Vertical)", value=True)
                    filter5 = st.checkbox("Sharpen", value=True)
                
                with col3:
                    filter3 = st.checkbox("Ridge Detection", value=False)
                    filter6 = st.checkbox("Identity", value=False)
                
                # Create a list of selected filters
                filters_to_show = []
                if filter1:
                    filters_to_show.append("Horizontal Edge Detection")
                if filter2:
                    filters_to_show.append("Vertical Edge Detection")
                if filter3:
                    filters_to_show.append("Ridge Detection")
                if filter4:
                    filters_to_show.append("Gaussian Blur")
                if filter5:
                    filters_to_show.append("Sharpen")
                if filter6:
                    filters_to_show.append("Identity")
                
                # Make sure at least one filter is selected
                if not filters_to_show:
                    st.warning("Please select at least one filter.")
                    filters_to_show = ["Horizontal Edge Detection"]
                
                # Activation function selection
                selected_activation = st.selectbox(
                    "Select Activation Function:",
                    ["ReLU", "Sigmoid", "Tanh"]
                )
                
                # Pooling operation selection
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_pooling = st.selectbox(
                        "Select Pooling Operation:",
                        ["Max Pooling", "Average Pooling"]
                    )
                
                with col2:
                    pooling_size = st.slider("Pooling Size:", min_value=2, max_value=4, value=2)
                
                # Create a dictionary of selected filters
                selected_filters = {name: filters.COMMON_FILTERS[name] for name in filters_to_show}
                
                # Visualize complete CNN process
                results = visualize_complete_cnn_process(
                    processed_img, 
                    selected_filters, 
                    selected_activation, 
                    selected_pooling, 
                    pooling_size, 
                    title="Complete CNN Processing for Sample Image"
                )
                
                # Display activation function comparison for one filter
                st.header("Activation Function Comparison")
                
                # Select a filter to compare activation functions
                activation_filter_name = st.selectbox("Select a filter to compare activation functions", filters_to_show)
                
                # Get the convolution result for the selected filter
                convolution_result = results["convolution"][activation_filter_name]
                
                # Visualize different activation functions
                visualize_different_activation_functions(convolution_result)
                
                # Display pooling operation comparison
                st.header("Pooling Operation Comparison")
                
                # Get the ReLU result for the selected filter
                relu_result = results["activation"][activation_filter_name]
                
                # Visualize different pooling operations
                visualize_different_pooling_operations(relu_result)

            # Upload Your Own Image section
            elif app_mode == "Upload Your Own Image":
                st.header("Upload Your Own Image")
                
                st.markdown("""
                Upload your own image to see how a CNN would process it through different layers.
                """)
                
                # File uploader
                uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
                
                if uploaded_file is not None:
                    try:
                        # Load the image
                        image = Image.open(uploaded_file).convert('RGB')
                        
                        # Display original image
                        st.subheader("Original Image")
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                        
                        # Process image (convert to grayscale but maintain original size)
                        processed_img = preprocess_image(image)
                        
                        # CNN Processing Options
                        st.subheader("CNN Layer Settings")
                        
                        # Filter selection
                        st.markdown("**Select filters to apply:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            filter1 = st.checkbox("Edge Detection (Horizontal)", value=True)
                            filter4 = st.checkbox("Gaussian Blur", value=False)
                        
                        with col2:
                            filter2 = st.checkbox("Edge Detection (Vertical)", value=True)
                            filter5 = st.checkbox("Sharpen", value=True)
                        
                        with col3:
                            filter3 = st.checkbox("Ridge Detection", value=False)
                            filter6 = st.checkbox("Identity", value=False)
                        
                        # Create a list of selected filters
                        filters_to_show = []
                        if filter1:
                            filters_to_show.append("Horizontal Edge Detection")
                        if filter2:
                            filters_to_show.append("Vertical Edge Detection")
                        if filter3:
                            filters_to_show.append("Ridge Detection")
                        if filter4:
                            filters_to_show.append("Gaussian Blur")
                        if filter5:
                            filters_to_show.append("Sharpen")
                        if filter6:
                            filters_to_show.append("Identity")
                        
                        # Make sure at least one filter is selected
                        if not filters_to_show:
                            st.warning("Please select at least one filter.")
                            filters_to_show = ["Horizontal Edge Detection"]
                        
                        # Activation function selection
                        selected_activation = st.selectbox(
                            "Select Activation Function:",
                            ["ReLU", "Sigmoid", "Tanh"]
                        )
                        
                        # Pooling operation selection
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            selected_pooling = st.selectbox(
                                "Select Pooling Operation:",
                                ["Max Pooling", "Average Pooling"]
                            )
                        
                        with col2:
                            pooling_size = st.slider("Pooling Size:", min_value=2, max_value=4, value=2)
                        
                        # Create a dictionary of selected filters
                        selected_filters = {name: filters.COMMON_FILTERS[name] for name in filters_to_show}
                        
                        # Visualize complete CNN process
                        results = visualize_complete_cnn_process(
                            processed_img, 
                            selected_filters, 
                            selected_activation, 
                            selected_pooling, 
                            pooling_size, 
                            title="Complete CNN Processing for Your Image"
                        )
                        
                        # Display activation function comparison for one filter
                        st.header("Activation Function Comparison")
                        
                        # Select a filter to compare activation functions
                        activation_filter_name = st.selectbox("Select a filter to compare activation functions", filters_to_show)
                        
                        # Get the convolution result for the selected filter
                        convolution_result = results["convolution"][activation_filter_name]
                        
                        # Visualize different activation functions
                        visualize_different_activation_functions(convolution_result)
                        
                        # Display pooling operation comparison
                        st.header("Pooling Operation Comparison")
                        
                        # Get the ReLU result for the selected filter
                        relu_result = results["activation"][activation_filter_name]
                        
                        # Visualize different pooling operations
                        visualize_different_pooling_operations(relu_result)
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("Please try uploading a different image.")

            # Custom Filter Builder section
            elif app_mode == "Custom Filter Builder":
                st.header("Custom Filter Builder")
                
                st.markdown("""
                Create your own convolutional filter and test it on sample images.
                This tool helps you understand how different filter designs extract different features.
                """)
                
                # Generate and display the custom filter
                custom_filter = generate_custom_filter()