import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import torch
import torch.nn.functional as F

from utils import convert_tf_to_numpy, convert_torch_to_numpy

def display_cnn_architecture():
    """Display basic CNN architecture diagram"""
    # Create a simple CNN architecture diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Define components and their positions
    components = [
        {'name': 'Input\n8x8x1', 'pos': 0, 'width': 0.5, 'color': 'lightblue'},
        {'name': 'Conv\n6x6x16', 'pos': 1.5, 'width': 0.5, 'color': 'lightgreen'},
        {'name': 'ReLU\n6x6x16', 'pos': 3, 'width': 0.5, 'color': 'pink'},
        {'name': 'Pool\n3x3x16', 'pos': 4.5, 'width': 0.5, 'color': 'lightyellow'},
        {'name': 'Flatten\n144', 'pos': 6, 'width': 0.5, 'color': 'lightgray'},
        {'name': 'FC\n10', 'pos': 7.5, 'width': 0.5, 'color': 'orange'},
        {'name': 'Softmax\n10', 'pos': 9, 'width': 0.5, 'color': 'lightcoral'}
    ]
    
    # Plot components
    for comp in components:
        ax.add_patch(plt.Rectangle((comp['pos'], 0.5), comp['width'], 1, 
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

def explain_convolution_operation():
    """Explain convolution operation with visual representation"""
    # Create visualization for convolution operation
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Example input (4x4)
    input_data = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    
    # Example filter (3x3)
    filter_data = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    # Resulting output (2x2)
    output_data = np.array([
        [4, 3],
        [3, 4]
    ])
    
    # Plot input
    axs[0].imshow(input_data, cmap='Blues')
    axs[0].set_title('Input (4x4)')
    for i in range(4):
        for j in range(4):
            axs[0].text(j, i, str(input_data[i, j]), ha='center', va='center', color='black')
    
    # Plot filter
    axs[1].imshow(filter_data, cmap='Greens')
    axs[1].set_title('Filter (3x3)')
    for i in range(3):
        for j in range(3):
            axs[1].text(j, i, str(filter_data[i, j]), ha='center', va='center', color='black')
    
    # Plot output
    axs[2].imshow(output_data, cmap='Reds')
    axs[2].set_title('Output (2x2)')
    for i in range(2):
        for j in range(2):
            axs[2].text(j, i, str(output_data[i, j]), ha='center', va='center', color='black')
    
    # Remove ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Explain the calculation
    st.markdown("""
    ### How Convolution Works:
    
    1. **Slide the filter** over the input image
    2. **Multiply** element-wise and **sum** the results
    3. **Store** the sum in the corresponding position in the output
    
    #### Example calculation for the top-left output:
    ```
    0*1 + 1*0 + 1*1 +
    0*0 + 1*1 + 0*0 +
    1*1 + 0*0 + 1*1 = 4
    ```
    """)

def explain_pooling_operation():
    """Explain pooling operation with visual representation"""
    # Create visualization for pooling operation
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # Example input (4x4)
    input_data = np.array([
        [1, 2, 5, 3],
        [8, 4, 6, 9],
        [3, 7, 1, 2],
        [5, 6, 4, 8]
    ])
    
    # Resulting output (2x2) with max pooling
    output_data = np.array([
        [8, 9],
        [7, 8]
    ])
    
    # Plot input
    axs[0].imshow(input_data, cmap='Blues')
    axs[0].set_title('Input (4x4)')
    
    # Draw grid to show pooling regions
    axs[0].axvline(x=1.5, color='red', linewidth=2)
    axs[0].axhline(y=1.5, color='red', linewidth=2)
    
    # Add values
    for i in range(4):
        for j in range(4):
            axs[0].text(j, i, str(input_data[i, j]), ha='center', va='center', color='black')
    
    # Plot output
    axs[1].imshow(output_data, cmap='Reds')
    axs[1].set_title('Max Pooling Output (2x2)')
    
    # Add values
    for i in range(2):
        for j in range(2):
            axs[1].text(j, i, str(output_data[i, j]), ha='center', va='center', color='black')
    
    # Remove ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Explain the calculation
    st.markdown("""
    ### How Max Pooling Works:
    
    1. **Divide** the input into regions (typically 2x2)
    2. **Take the maximum value** from each region
    3. **Output** these maximum values
    
    #### Example calculation:
    - Top-left region (1, 2, 8, 4): Max = 8
    - Top-right region (5, 3, 6, 9): Max = 9
    - Bottom-left region (3, 7, 5, 6): Max = 7
    - Bottom-right region (1, 2, 4, 8): Max = 8
    """)

def explain_activation_functions():
    """Explain activation functions with plots"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Generate x values
    x = np.linspace(-5, 5, 100)
    
    # ReLU function
    relu = np.maximum(0, x)
    
    # Softmax function (for visualization)
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))
    
    # Example of softmax input
    softmax_in = np.array([-2, -1, 0, 1, 2])
    softmax_out = softmax(softmax_in)
    
    # Plot ReLU
    axs[0].plot(x, relu, 'b-', linewidth=2)
    axs[0].set_title('ReLU Activation')
    axs[0].set_xlabel('Input')
    axs[0].set_ylabel('Output')
    axs[0].grid(True)
    axs[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot Softmax
    axs[1].bar(range(len(softmax_in)), softmax_out, tick_label=softmax_in)
    axs[1].set_title('Softmax Activation')
    axs[1].set_xlabel('Input Values')
    axs[1].set_ylabel('Output Probabilities')
    
    # Display plots
    plt.tight_layout()
    st.pyplot(fig)
    
    # Explain activation functions
    st.markdown("""
    ### Activation Functions:
    
    #### ReLU (Rectified Linear Unit):
    - Introduces **non-linearity** into the network
    - Simple formula: **f(x) = max(0, x)**
    - Outputs the input directly if positive, otherwise outputs zero
    - Helps with vanishing gradient problem
    
    #### Softmax:
    - Converts raw scores to **probabilities** (used in output layer)
    - Formula: **softmax(x_i) = exp(x_i) / sum(exp(x_j))**
    - Ensures all outputs sum to 1.0
    - The larger the input value, the higher its corresponding probability
    """)

def display_scratch_cnn_architecture(model):
    """Display CNN architecture from scratch implementation"""
    # Create visualization of the architecture
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define layers
    layers = [
        {'name': 'Input', 'shape': f"{model.input_shape[0]}x{model.input_shape[1]}x{model.input_shape[2]}", 'color': 'lightblue'},
        {'name': 'Conv', 'shape': f"{model.conv1_output_shape[0]}x{model.conv1_output_shape[1]}x{model.conv_filters}", 'color': 'lightgreen'},
        {'name': 'ReLU', 'shape': f"{model.conv1_output_shape[0]}x{model.conv1_output_shape[1]}x{model.conv_filters}", 'color': 'pink'},
        {'name': 'Pool', 'shape': f"{model.pool1_output_shape[0]}x{model.pool1_output_shape[1]}x{model.conv_filters}", 'color': 'lightyellow'},
        {'name': 'Flatten', 'shape': f"{model.flatten_size}", 'color': 'lightgray'},
        {'name': 'FC', 'shape': "10", 'color': 'orange'},
        {'name': 'Softmax', 'shape': "10", 'color': 'lightcoral'}
    ]
    
    # Plot layers
    for i, layer in enumerate(layers):
        y_pos = 0.5
        height = 1
        width = 0.6
        x_pos = i * 1.5
        
        # Draw rectangle for layer
        rect = plt.Rectangle((x_pos, y_pos), width, height, facecolor=layer['color'], edgecolor='black')
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_pos + width/2, y_pos + height/2, f"{layer['name']}\n{layer['shape']}", 
                ha='center', va='center', fontsize=9)
        
        # Add arrow to next layer
        if i < len(layers) - 1:
            ax.arrow(x_pos + width, y_pos + height/2, 0.8, 0, 
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, (len(layers) - 1) * 1.5 + 1.2)
    ax.set_ylim(0, 2)
    ax.set_axis_off()
    
    # Add legend for parameters
    param_text = (
        f"Total Parameters:\n"
        f"Conv Filters: {model.conv_filters} (3x3)\n"
        f"Conv Params: {3*3*model.input_shape[2]*model.conv_filters + model.conv_filters}\n"
        f"FC Params: {model.flatten_size * 10 + 10}\n"
    )
    ax.text(0, -0.2, param_text, fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

def visualize_convolution_scratch(model, input_image):
    """Visualize convolution operation in scratch CNN"""
    # Perform convolution
    conv_output = model.convolution(input_image, model.conv1_filters, model.conv1_bias)
    
    # Create visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show input image
    axs[0].imshow(input_image.reshape(8, 8), cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    # Show sample filters (display first 3)
    filter_idx = 0
    filter_img = model.conv1_filters[:, :, 0, filter_idx]
    axs[1].imshow(filter_img, cmap='viridis')
    axs[1].set_title(f'Filter {filter_idx}')
    axs[1].axis('off')
    
    # Show output feature map (for the filter shown)
    axs[2].imshow(conv_output[:, :, filter_idx], cmap='inferno')
    axs[2].set_title(f'Feature Map {filter_idx}')
    axs[2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add slider to view different filters
    selected_filter = st.slider("Select Filter", 0, model.conv_filters-1, 0)
    
    # Show selected filter and its feature map
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # Show selected filter
    filter_img = model.conv1_filters[:, :, 0, selected_filter]
    axs[0].imshow(filter_img, cmap='viridis')
    axs[0].set_title(f'Filter {selected_filter}')
    axs[0].axis('off')
    
    # Show corresponding feature map
    axs[1].imshow(conv_output[:, :, selected_filter], cmap='inferno')
    axs[1].set_title(f'Feature Map {selected_filter}')
    axs[1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def visualize_pooling_scratch(model, input_image):
    """Visualize pooling operation in scratch CNN"""
    # Perform convolution and activation
    conv_output = model.convolution(input_image, model.conv1_filters, model.conv1_bias)
    relu_output = model.relu(conv_output)
    
    # Perform pooling
    pool_output = model.max_pooling(relu_output)
    
    # Select a filter to visualize
    filter_idx = 0
    
    # Create visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show post-activation feature map
    axs[0].imshow(relu_output[:, :, filter_idx], cmap='inferno')
    axs[0].set_title(f'Post-ReLU Feature Map\n{relu_output.shape[0]}x{relu_output.shape[1]}')
    axs[0].axis('off')
    
    # Show pooling grid
    axs[1].imshow(relu_output[:, :, filter_idx], cmap='inferno')
    axs[1].set_title('Pooling Regions (2x2)')
    axs[1].axis('off')
    
    # Add grid lines to show pooling regions
    for i in range(0, relu_output.shape[0], 2):
        axs[1].axhline(y=i-0.5, color='white', linestyle='-', linewidth=1)
    for j in range(0, relu_output.shape[1], 2):
        axs[1].axvline(x=j-0.5, color='white', linestyle='-', linewidth=1)
    
    # Show pooled output
    axs[2].imshow(pool_output[:, :, filter_idx], cmap='inferno')
    axs[2].set_title(f'Pooled Feature Map\n{pool_output.shape[0]}x{pool_output.shape[1]}')
    axs[2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add slider to view different filters
    selected_filter = st.slider("Select Filter for Pooling", 0, model.conv_filters-1, 0, key="pool_filter")
    
    # Show selected filter's feature maps before and after pooling
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # Show selected feature map before pooling
    axs[0].imshow(relu_output[:, :, selected_filter], cmap='inferno')
    axs[0].set_title(f'Before Pooling\n{relu_output.shape[0]}x{relu_output.shape[1]}')
    axs[0].axis('off')
    
    # Add grid lines to show pooling regions
    for i in range(0, relu_output.shape[0], 2):
        axs[0].axhline(y=i-0.5, color='white', linestyle='-', linewidth=1)
    for j in range(0, relu_output.shape[1], 2):
        axs[0].axvline(x=j-0.5, color='white', linestyle='-', linewidth=1)
    
    # Show selected feature map after pooling
    axs[1].imshow(pool_output[:, :, selected_filter], cmap='inferno')
    axs[1].set_title(f'After Pooling\n{pool_output.shape[0]}x{pool_output.shape[1]}')
    axs[1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def explain_backpropagation_scratch():
    """Explain backpropagation in CNN from scratch"""
    st.markdown("""
    ### Backpropagation in CNN
    
    Backpropagation is the algorithm used to train the CNN by updating the weights based on the error.
    
    #### The process follows these steps:
    
    1. **Forward Pass**:
       - Input data flows through the network
       - Activations at each layer are stored
    
    2. **Error Calculation**:
       - Compare network output with desired output
       - Calculate loss (error)
    
    3. **Backward Pass**:
       - Calculate error gradient for output layer
       - Propagate error gradients backward through the network
       - Update weights based on gradients and learning rate
    
    #### Gradients Flow in CNN:
    
    - **Fully Connected Layer**: Similar to standard neural networks
    - **Pooling Layer**: Gradients are routed only to the neurons that had maximum value
    - **Convolution Layer**: Complex gradient calculation involving convolution operations
    
    The key challenge in CNNs is calculating gradients for shared weights in convolution filters.
    """)
    
    # Visualization of gradient flow
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define components and their positions
    components = [
        {'name': 'Input\n8x8x1', 'pos': 0, 'width': 0.5, 'height': 1, 'color': 'lightblue'},
        {'name': 'Conv\n6x6x16', 'pos': 1.5, 'width': 0.5, 'height': 1, 'color': 'lightgreen'},
        {'name': 'Pool\n3x3x16', 'pos': 3, 'width': 0.5, 'height': 1, 'color': 'lightyellow'},
        {'name': 'FC\n10', 'pos': 4.5, 'width': 0.5, 'height': 1, 'color': 'orange'},
        {'name': 'Output\n10', 'pos': 6, 'width': 0.5, 'height': 1, 'color': 'lightcoral'}
    ]
    
    # Plot components
    for comp in components:
        ax.add_patch(plt.Rectangle((comp['pos'], 1), comp['width'], comp['height'], 
                                  facecolor=comp['color'], edgecolor='black'))
        ax.text(comp['pos'] + comp['width']/2, 1.5, comp['name'], 
               ha='center', va='center', fontsize=9)
    
    # Add forward arrows
    for i in range(len(components)-1):
        start_x = components[i]['pos'] + components[i]['width']
        end_x = components[i+1]['pos']
        ax.arrow(start_x, 1.5, end_x - start_x - 0.05, 0, 
                head_width=0.07, head_length=0.05, fc='blue', ec='blue')
    
    # Add backward arrows (gradient flow)
    for i in range(len(components)-1, 0, -1):
        start_x = components[i]['pos']
        end_x = components[i-1]['pos'] + components[i-1]['width']
        ax.arrow(start_x, 1.2, end_x - start_x + 0.05, 0, 
                head_width=0.07, head_length=0.05, fc='red', ec='red')
    
    # Add error calculation
    ax.add_patch(plt.Rectangle((6.5, 1), 1, 1, facecolor='lightpink', edgecolor='black'))
    ax.text(7, 1.5, 'Loss\nCalculation', ha='center', va='center', fontsize=9)
    
    # Add connection to loss
    ax.arrow(6.5, 1.5, -0.5, 0, head_width=0.07, head_length=0.05, fc='red', ec='red')
    ax.arrow(6 + 0.25, 1.1, 0.5, 0, head_width=0.07, head_length=0.05, fc='black', ec='black')
    
    # Add legend
    ax.add_patch(plt.Rectangle((1.5, 0.2), 0.5, 0.2, facecolor='blue', edgecolor='blue'))
    ax.text(2.1, 0.3, 'Forward Pass', ha='left', va='center', fontsize=8)
    
    ax.add_patch(plt.Rectangle((3.5, 0.2), 0.5, 0.2, facecolor='red', edgecolor='red'))
    ax.text(4.1, 0.3, 'Backward Pass (Gradients)', ha='left', va='center', fontsize=8)
    
    # Set axis limits and turn off axis
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    
    st.pyplot(fig)

def show_scratch_filters(model):
    """Show filters from scratch CNN model"""
    # Display filters
    fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    axs = axs.flatten()
    
    # Plot filters (up to 16)
    num_filters = min(model.conv_filters, 16)
    for i in range(num_filters):
        if i < len(axs):
            filter_img = model.conv1_filters[:, :, 0, i]
            axs[i].imshow(filter_img, cmap='viridis')
            axs[i].set_title(f'Filter {i}')
            axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_scratch_feature_maps(model, input_image):
    """Show feature maps from scratch CNN model"""
    # Forward pass to get feature maps
    conv_output = model.convolution(input_image, model.conv1_filters, model.conv1_bias)
    relu_output = model.relu(conv_output)
    pool_output = model.max_pooling(relu_output)
    
    # Display input image
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(input_image.reshape(8, 8), cmap='gray')
    ax.set_title('Input Image')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display feature maps after convolution
    st.subheader("Feature Maps after Convolution+ReLU")
    
    # Split into multiple rows for display
    n_cols = 4
    n_rows = (model.conv_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(model.conv_filters):
        if i < len(axs):
            axs[i].imshow(relu_output[:, :, i], cmap='inferno')
            axs[i].set_title(f'Filter {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(model.conv_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display feature maps after pooling
    st.subheader("Feature Maps after Pooling")
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(model.conv_filters):
        if i < len(axs):
            axs[i].imshow(pool_output[:, :, i], cmap='inferno')
            axs[i].set_title(f'Filter {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(model.conv_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def visualize_tf_layers(model, input_data):
    """Visualize TensorFlow CNN layers"""
    # Create models to extract intermediates
    conv_model = tf.keras.models.Model(
        inputs=model.model.inputs,
        outputs=model.model.layers[0].output
    )
    
    pool_model = tf.keras.models.Model(
        inputs=model.model.inputs,
        outputs=model.model.layers[1].output
    )
    
    # Get outputs
    conv_output = conv_model.predict(input_data)
    pool_output = pool_model.predict(input_data)
    
    # Visualize layer outputs
    st.subheader("Convolutional Layer Output")
    
    # Display filters
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(input_data[0, :, :, 0], cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    st.pyplot(fig)
    
    # Display Conv layer output
    n_filters = conv_output.shape[-1]
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(conv_output[0, :, :, i], cmap='inferno')
            axs[i].set_title(f'Feature Map {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display Pooling layer output
    st.subheader("Pooling Layer Output")
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(pool_output[0, :, :, i], cmap='inferno')
            axs[i].set_title(f'Pooled Map {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_tf_feature_maps(model, input_image):
    """Show feature maps from TensorFlow CNN model"""
    # Reshape input for model
    input_reshaped = input_image.reshape(1, 8, 8, 1)
    
    # Get feature maps
    feature_maps = model.get_feature_maps(input_reshaped)
    
    # Display input image
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(input_image.reshape(8, 8), cmap='gray')
    ax.set_title('Input Image')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display feature maps
    st.subheader("Feature Maps")
    
    # Split into multiple rows for display
    n_filters = feature_maps.shape[-1]
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(feature_maps[:, :, i], cmap='inferno')
            axs[i].set_title(f'Feature Map {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_tf_filters(model):
    """Show filters from TensorFlow CNN model"""
    # Get filters
    filters = model.get_filters()
    
    if filters is not None:
        # Get dimensions
        filter_height, filter_width, in_channels, n_filters = filters.shape
        
        # Display filters
        st.subheader("Convolutional Filters")
        
        # Split into multiple rows for display
        n_cols = 4
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        axs = axs.flatten()
        
        for i in range(n_filters):
            if i < len(axs):
                # For RGB images we'd need to handle 3 channels, but here we have 1 channel
                axs[i].imshow(filters[:, :, 0, i], cmap='viridis')
                axs[i].set_title(f'Filter {i}')
                axs[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_filters, len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No convolutional filters found in the model.")

def visualize_torch_layers(model, input_image):
    """Visualize PyTorch CNN layers"""
    # Create tensor from input
    input_tensor = torch.tensor(input_image.reshape(1, 1, 8, 8), dtype=torch.float32)
    
    # Get model
    torch_model = model.model
    
    # Set model to eval mode
    torch_model.eval()
    
    # Get layer outputs
    with torch.no_grad():
        # Convolutional layer output
        conv_output = torch_model.conv1(input_tensor)
        
        # ReLU activation
        relu_output = F.relu(conv_output)
        
        # Pooling layer output
        pool_output = torch_model.pool(relu_output)
    
    # Convert outputs to numpy
    conv_np = convert_torch_to_numpy(conv_output)[0]
    relu_np = convert_torch_to_numpy(relu_output)[0]
    pool_np = convert_torch_to_numpy(pool_output)[0]
    
    # Display original image
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(input_image.reshape(8, 8), cmap='gray')
    ax.set_title('Input Image')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display Conv layer output
    st.subheader("Convolutional Layer Output")
    
    n_filters = conv_np.shape[0]
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(conv_np[i], cmap='inferno')
            axs[i].set_title(f'Feature Map {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display Pooling layer output
    st.subheader("Pooling Layer Output")
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(pool_np[i], cmap='inferno')
            axs[i].set_title(f'Pooled Map {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_torch_feature_maps(model, input_image):
    """Show feature maps from PyTorch CNN model"""
    # Get feature maps
    feature_maps = model.get_feature_maps(input_image)
    
    # Display input image
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(input_image.reshape(8, 8), cmap='gray')
    ax.set_title('Input Image')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display feature maps
    st.subheader("Feature Maps")
    
    # Split into multiple rows for display
    n_filters = feature_maps.shape[0]
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(feature_maps[i], cmap='inferno')
            axs[i].set_title(f'Feature Map {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_torch_filters(filters):
    """Plot filters from PyTorch model"""
    # Create figure
    n_filters = filters.shape[0]
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_filters):
        if i < len(axs):
            # For grayscale images, we have 1 input channel
            axs[i].imshow(filters[i, 0], cmap='viridis')
            axs[i].set_title(f'Filter {i}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig
