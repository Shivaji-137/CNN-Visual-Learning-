import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import networkx as nx
import io
import time
import math
import re

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# Available layer types for architecture building
LAYER_TYPES = {
    "Convolutional": {
        "pytorch": nn.Conv2d,
        "tensorflow": tf.keras.layers.Conv2D,
        "params": {
            "filters": {"type": "int", "min": 1, "max": 256, "default": 32},
            "kernel_size": {"type": "int", "min": 1, "max": 7, "default": 3},
            "stride": {"type": "int", "min": 1, "max": 3, "default": 1},
            "padding": {"type": "select", "options": ["valid", "same"], "default": "same"}
        },
        "short_name": "Conv"
    },
    "MaxPooling": {
        "pytorch": nn.MaxPool2d,
        "tensorflow": tf.keras.layers.MaxPooling2D,
        "params": {
            "pool_size": {"type": "int", "min": 1, "max": 4, "default": 2},
            "stride": {"type": "int", "min": 1, "max": 4, "default": 2}
        },
        "short_name": "MaxPool"
    },
    "AveragePooling": {
        "pytorch": nn.AvgPool2d,
        "tensorflow": tf.keras.layers.AveragePooling2D,
        "params": {
            "pool_size": {"type": "int", "min": 1, "max": 4, "default": 2},
            "stride": {"type": "int", "min": 1, "max": 4, "default": 2}
        },
        "short_name": "AvgPool"
    },
    "ReLU": {
        "pytorch": nn.ReLU,
        "tensorflow": "relu",  # Activation name for TF
        "params": {},
        "short_name": "ReLU"
    },
    "Sigmoid": {
        "pytorch": nn.Sigmoid,
        "tensorflow": "sigmoid",  # Activation name for TF
        "params": {},
        "short_name": "Sigmoid"
    },
    "Tanh": {
        "pytorch": nn.Tanh,
        "tensorflow": "tanh",  # Activation name for TF
        "params": {},
        "short_name": "Tanh"
    },
    "LeakyReLU": {
        "pytorch": nn.LeakyReLU,
        "tensorflow": tf.keras.layers.LeakyReLU,
        "params": {
            "alpha": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1}
        },
        "short_name": "LReLU"
    },
    "BatchNormalization": {
        "pytorch": nn.BatchNorm2d,
        "tensorflow": tf.keras.layers.BatchNormalization,
        "params": {},
        "short_name": "BN"
    },
    "Dropout": {
        "pytorch": nn.Dropout,
        "tensorflow": tf.keras.layers.Dropout,
        "params": {
            "rate": {"type": "float", "min": 0.0, "max": 0.9, "default": 0.5}
        },
        "short_name": "Drop"
    },
    "Flatten": {
        "pytorch": nn.Flatten,
        "tensorflow": tf.keras.layers.Flatten,
        "params": {},
        "short_name": "Flat"
    },
    "FullyConnected": {
        "pytorch": nn.Linear,
        "tensorflow": tf.keras.layers.Dense,
        "params": {
            "units": {"type": "int", "min": 1, "max": 1024, "default": 128},
            "activation": {"type": "select", "options": ["none", "relu", "sigmoid", "tanh"], "default": "relu"}
        },
        "short_name": "FC"
    }
}

def generate_pytorch_code(layers, input_shape):
    """Generate PyTorch code for the network architecture"""
    code_lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        "class CNN(nn.Module):",
        "    def __init__(self):",
        "        super(CNN, self).__init__()",
        ""
    ]
    
    # Track shapes through the network
    current_shape = list(input_shape)  # Copy the input shape
    in_channels = current_shape[0]
    
    # Keep track of layer definitions
    layer_defs = []
    forward_ops = []
    
    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        params = layer["params"]
        layer_info = LAYER_TYPES[layer_type]
        
        # Handle convolutional layers
        if layer_type == "Convolutional":
            filters = params.get("filters", 32)
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", "same")
            
            # Convert padding to integer
            padding_val = kernel_size // 2 if padding == "same" else 0
            
            layer_name = f"conv{i+1}"
            layer_def = f"        self.{layer_name} = nn.Conv2d({in_channels}, {filters}, kernel_size={kernel_size}, stride={stride}, padding={padding_val})"
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
            
            # Update shape
            if padding == "valid":
                current_shape[1] = math.floor((current_shape[1] - kernel_size + 1) / stride)
                current_shape[2] = math.floor((current_shape[2] - kernel_size + 1) / stride)
            else:  # same padding
                current_shape[1] = math.ceil(current_shape[1] / stride)
                current_shape[2] = math.ceil(current_shape[2] / stride)
            
            # Update channels
            current_shape[0] = filters
            in_channels = filters
        
        # Handle pooling layers
        elif layer_type in ["MaxPooling", "AveragePooling"]:
            pool_size = params.get("pool_size", 2)
            stride = params.get("stride", 2)
            
            layer_name = f"pool{i+1}"
            if layer_type == "MaxPooling":
                layer_def = f"        self.{layer_name} = nn.MaxPool2d(kernel_size={pool_size}, stride={stride})"
            else:
                layer_def = f"        self.{layer_name} = nn.AvgPool2d(kernel_size={pool_size}, stride={stride})"
            
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
            
            # Update shape
            current_shape[1] = math.floor((current_shape[1] - pool_size) / stride + 1)
            current_shape[2] = math.floor((current_shape[2] - pool_size) / stride + 1)
        
        # Handle activation functions
        elif layer_type in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]:
            layer_name = f"{layer_type.lower()}{i+1}"
            
            if layer_type == "LeakyReLU":
                alpha = params.get("alpha", 0.1)
                layer_def = f"        self.{layer_name} = nn.LeakyReLU(negative_slope={alpha})"
            else:
                layer_def = f"        self.{layer_name} = nn.{layer_type}()"
            
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
        
        # Handle batch normalization
        elif layer_type == "BatchNormalization":
            layer_name = f"bn{i+1}"
            layer_def = f"        self.{layer_name} = nn.BatchNorm2d({in_channels})"
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
        
        # Handle dropout
        elif layer_type == "Dropout":
            rate = params.get("rate", 0.5)
            layer_name = f"dropout{i+1}"
            layer_def = f"        self.{layer_name} = nn.Dropout(p={rate})"
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
        
        # Handle flatten
        elif layer_type == "Flatten":
            layer_name = "flatten"
            layer_def = f"        self.{layer_name} = nn.Flatten()"
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
            
            # Update shape - flatten to 1D
            flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
            current_shape = [flattened_size]
        
        # Handle fully connected layers
        elif layer_type == "FullyConnected":
            units = params.get("units", 128)
            activation = params.get("activation", "relu")
            
            # If this is the first FC layer after conv/pool, we need to know the flattened size
            if len(current_shape) > 1:
                flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
                layer_defs.append(f"        self.flatten = nn.Flatten()")
                forward_ops.append(f"x = self.flatten(x)")
                current_shape = [flattened_size]
            
            layer_name = f"fc{i+1}"
            layer_def = f"        self.{layer_name} = nn.Linear({current_shape[0]}, {units})"
            layer_defs.append(layer_def)
            forward_ops.append(f"x = self.{layer_name}(x)")
            
            # Add activation if specified
            if activation != "none":
                act_name = f"{activation}{i+1}"
                act_class = "ReLU" if activation == "relu" else "Sigmoid" if activation == "sigmoid" else "Tanh"
                layer_defs.append(f"        self.{act_name} = nn.{act_class}()")
                forward_ops.append(f"x = self.{act_name}(x)")
            
            # Update shape
            current_shape = [units]
    
    # Add the layer definitions to the code
    code_lines.extend(layer_defs)
    
    # Add the forward method
    code_lines.append("")
    code_lines.append("    def forward(self, x):")
    
    # Add forward operations
    for op in forward_ops:
        code_lines.append(f"        {op}")
    
    # Return the final output
    code_lines.append("        return x")
    code_lines.append("")
    
    # Model instantiation example
    code_lines.append("# Example usage:")
    code_lines.append(f"model = CNN()")
    input_channels, input_height, input_width = input_shape
    code_lines.append(f"x = torch.randn(1, {input_channels}, {input_height}, {input_width})  # Batch size of 1")
    code_lines.append("output = model(x)")
    code_lines.append(f"print(output.shape)  # Expected shape based on architecture")
    
    return "\n".join(code_lines)

def generate_tensorflow_code(layers, input_shape):
    """Generate TensorFlow code for the network architecture"""
    code_lines = [
        "import tensorflow as tf",
        "from tensorflow.keras import layers, models",
        "",
        "def create_model():",
        f"    # Input shape: {input_shape}",
        f"    inputs = tf.keras.Input(shape=({input_shape[1]}, {input_shape[2]}, {input_shape[0]}))",
        "    x = inputs"
    ]
    
    # Track shapes through the network
    current_shape = list(input_shape)
    # Convert from channels_first to channels_last
    current_shape = [current_shape[1], current_shape[2], current_shape[0]]
    
    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        params = layer["params"]
        
        # Handle convolutional layers
        if layer_type == "Convolutional":
            filters = params.get("filters", 32)
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", "same")
            
            code_lines.append(f"    x = layers.Conv2D({filters}, kernel_size={kernel_size}, strides={stride}, padding='{padding}')(x)")
            
            # Update shape
            if padding == "valid":
                current_shape[0] = math.floor((current_shape[0] - kernel_size) / stride + 1)
                current_shape[1] = math.floor((current_shape[1] - kernel_size) / stride + 1)
            else:  # same padding
                current_shape[0] = math.ceil(current_shape[0] / stride)
                current_shape[1] = math.ceil(current_shape[1] / stride)
            current_shape[2] = filters
        
        # Handle pooling layers
        elif layer_type in ["MaxPooling", "AveragePooling"]:
            pool_size = params.get("pool_size", 2)
            stride = params.get("stride", 2)
            
            if layer_type == "MaxPooling":
                code_lines.append(f"    x = layers.MaxPooling2D(pool_size={pool_size}, strides={stride})(x)")
            else:
                code_lines.append(f"    x = layers.AveragePooling2D(pool_size={pool_size}, strides={stride})(x)")
            
            # Update shape
            current_shape[0] = math.floor((current_shape[0] - pool_size) / stride + 1)
            current_shape[1] = math.floor((current_shape[1] - pool_size) / stride + 1)
        
        # Handle activation functions
        elif layer_type in ["ReLU", "Sigmoid", "Tanh"]:
            activation = layer_type.lower()
            code_lines.append(f"    x = layers.Activation('{activation}')(x)")
        
        # Handle LeakyReLU
        elif layer_type == "LeakyReLU":
            alpha = params.get("alpha", 0.1)
            code_lines.append(f"    x = layers.LeakyReLU(alpha={alpha})(x)")
        
        # Handle batch normalization
        elif layer_type == "BatchNormalization":
            code_lines.append(f"    x = layers.BatchNormalization()(x)")
        
        # Handle dropout
        elif layer_type == "Dropout":
            rate = params.get("rate", 0.5)
            code_lines.append(f"    x = layers.Dropout(rate={rate})(x)")
        
        # Handle flatten
        elif layer_type == "Flatten":
            code_lines.append(f"    x = layers.Flatten()(x)")
            
            # Update shape - flatten to 1D
            flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
            current_shape = [flattened_size]
        
        # Handle fully connected layers
        elif layer_type == "FullyConnected":
            units = params.get("units", 128)
            activation = params.get("activation", "relu")
            
            if activation == "none":
                code_lines.append(f"    x = layers.Dense({units})(x)")
            else:
                code_lines.append(f"    x = layers.Dense({units}, activation='{activation}')(x)")
            
            # Update shape
            current_shape = [units]
    
    # Create the model
    code_lines.append("    model = tf.keras.Model(inputs=inputs, outputs=x)")
    code_lines.append("    return model")
    code_lines.append("")
    code_lines.append("# Create and summarize the model")
    code_lines.append("model = create_model()")
    code_lines.append("model.summary()")
    
    return "\n".join(code_lines)

def calculate_output_shape(layers, input_shape):
    """Calculate the output shape after each layer"""
    shapes = [list(input_shape)]  # Start with the input shape
    current_shape = list(input_shape)
    
    for layer in layers:
        layer_type = layer["type"]
        params = layer["params"]
        
        # Handle convolutional layers
        if layer_type == "Convolutional":
            filters = params.get("filters", 32)
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", "same")
            
            if padding == "valid":
                h_out = math.floor((current_shape[1] - kernel_size) / stride + 1)
                w_out = math.floor((current_shape[2] - kernel_size) / stride + 1)
            else:  # same padding
                h_out = math.ceil(current_shape[1] / stride)
                w_out = math.ceil(current_shape[2] / stride)
            
            current_shape = [filters, h_out, w_out]
        
        # Handle pooling layers
        elif layer_type in ["MaxPooling", "AveragePooling"]:
            pool_size = params.get("pool_size", 2)
            stride = params.get("stride", 2)
            
            h_out = math.floor((current_shape[1] - pool_size) / stride + 1)
            w_out = math.floor((current_shape[2] - pool_size) / stride + 1)
            current_shape = [current_shape[0], h_out, w_out]
        
        # Handle flatten layer
        elif layer_type == "Flatten":
            flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
            current_shape = [flattened_size]
        
        # Handle fully connected layers
        elif layer_type == "FullyConnected":
            units = params.get("units", 128)
            
            # If current shape is still 3D, flatten it first
            if len(current_shape) == 3:
                flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
                current_shape = [flattened_size]
            
            current_shape = [units]
        
        # Add the new shape
        shapes.append(list(current_shape))
    
    return shapes

def calculate_receptive_field(layers, input_shape):
    """Calculate the receptive field size at each layer"""
    receptive_fields = [1]  # Start with 1x1 at input
    jump = 1  # The distance between two adjacent units in the feature map
    receptive_field_size = 1  # Initial receptive field size
    
    for layer in layers:
        layer_type = layer["type"]
        params = layer["params"]
        
        # Only convolutional and pooling layers affect the receptive field
        if layer_type == "Convolutional":
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            
            # Update receptive field size
            receptive_field_size += (kernel_size - 1) * jump
            jump *= stride
        
        elif layer_type in ["MaxPooling", "AveragePooling"]:
            pool_size = params.get("pool_size", 2)
            stride = params.get("stride", 2)
            
            # Update receptive field size
            receptive_field_size += (pool_size - 1) * jump
            jump *= stride
        
        # Add the new receptive field size
        receptive_fields.append(receptive_field_size)
    
    return receptive_fields

def calculate_parameter_count(layers, input_shape):
    """Calculate the number of parameters for each layer"""
    param_counts = [0]  # No parameters for input
    total_params = 0
    
    # Track shapes through the network
    current_shape = list(input_shape)
    
    for layer in layers:
        layer_type = layer["type"]
        params = layer["params"]
        layer_params = 0
        
        # Handle convolutional layers
        if layer_type == "Convolutional":
            filters = params.get("filters", 32)
            kernel_size = params.get("kernel_size", 3)
            
            # Parameters = (kernel_h * kernel_w * in_channels + 1) * out_channels
            # +1 for bias per filter
            layer_params = (kernel_size * kernel_size * current_shape[0] + 1) * filters
            
            # Update current shape
            padding = params.get("padding", "same")
            stride = params.get("stride", 1)
            
            if padding == "valid":
                h_out = math.floor((current_shape[1] - kernel_size) / stride + 1)
                w_out = math.floor((current_shape[2] - kernel_size) / stride + 1)
            else:  # same padding
                h_out = math.ceil(current_shape[1] / stride)
                w_out = math.ceil(current_shape[2] / stride)
            
            current_shape = [filters, h_out, w_out]
        
        # Handle batch normalization
        elif layer_type == "BatchNormalization":
            # BatchNorm has 4 parameters per channel: gamma, beta, moving_mean, moving_variance
            layer_params = 4 * current_shape[0]
        
        # Handle pooling layers
        elif layer_type in ["MaxPooling", "AveragePooling"]:
            # Pooling layers have no parameters
            layer_params = 0
            
            # Update current shape
            pool_size = params.get("pool_size", 2)
            stride = params.get("stride", 2)
            
            h_out = math.floor((current_shape[1] - pool_size) / stride + 1)
            w_out = math.floor((current_shape[2] - pool_size) / stride + 1)
            current_shape = [current_shape[0], h_out, w_out]
        
        # Handle activation functions
        elif layer_type in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]:
            # Activation functions have no parameters
            layer_params = 0
        
        # Handle dropout
        elif layer_type == "Dropout":
            # Dropout has no parameters
            layer_params = 0
        
        # Handle flatten
        elif layer_type == "Flatten":
            # Flatten has no parameters
            layer_params = 0
            
            # Update current shape
            flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
            current_shape = [flattened_size]
        
        # Handle fully connected layers
        elif layer_type == "FullyConnected":
            units = params.get("units", 128)
            
            # If current shape is still 3D, flatten it first
            if len(current_shape) == 3:
                flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
                current_shape = [flattened_size]
            
            # Parameters = (in_features + 1) * out_features
            # +1 for bias per output unit
            layer_params = (current_shape[0] + 1) * units
            
            # Update current shape
            current_shape = [units]
        
        # Add the layer parameters
        param_counts.append(layer_params)
        total_params += layer_params
    
    return param_counts, total_params

def create_network_visualization(layers, input_shape, output_shapes, receptive_fields, param_counts):
    """Create a visualization of the network architecture"""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add input node
    G.add_node("Input", shape=input_shape, pos=(0, 0))
    
    # Track the layer positions
    layer_positions = {"Input": (0, 0)}
    x_spacing = 1
    
    # Add layer nodes and edges
    prev_node = "Input"
    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        params = layer["params"]
        short_name = LAYER_TYPES[layer_type]["short_name"]
        
        # Create a node label with parameters
        param_text = ""
        if layer_type == "Convolutional":
            filters = params.get("filters", 32)
            kernel_size = params.get("kernel_size", 3)
            param_text = f"{filters}@{kernel_size}x{kernel_size}"
        elif layer_type in ["MaxPooling", "AveragePooling"]:
            pool_size = params.get("pool_size", 2)
            param_text = f"{pool_size}x{pool_size}"
        elif layer_type == "FullyConnected":
            units = params.get("units", 128)
            param_text = f"{units}"
        
        # Create the node label
        if param_text:
            node_label = f"{short_name}\n{param_text}"
        else:
            node_label = short_name
        
        # Add the node with attributes
        node_name = f"{short_name}_{i+1}"
        output_shape = output_shapes[i+1]
        receptive_field = receptive_fields[i+1]
        params_count = param_counts[i+1]
        
        G.add_node(node_name, 
                  label=node_label, 
                  shape=output_shape, 
                  rf=receptive_field,
                  params=params_count)
        
        # Position the node (simple horizontal layout)
        x_pos = (i + 1) * x_spacing
        y_pos = 0
        layer_positions[node_name] = (x_pos, y_pos)
        
        # Add an edge from the previous node
        G.add_edge(prev_node, node_name)
        prev_node = node_name
    
    # Set node positions
    pos = layer_positions
    
    # Draw the network graph
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw nodes
    for node, (x, y) in pos.items():
        if node == "Input":
            shape = G.nodes[node]["shape"]
            label = f"Input\n{shape[0]}x{shape[1]}x{shape[2]}"
            plt.plot(x, y, 'o', markersize=20, color='lightblue')
            plt.text(x, y, label, ha='center', va='center', fontsize=8)
        else:
            shape = G.nodes[node]["shape"]
            rf = G.nodes[node]["rf"]
            params = G.nodes[node]["params"]
            label = G.nodes[node]["label"]
            
            # Format the shape for display
            if len(shape) == 3:
                shape_text = f"{shape[0]}x{shape[1]}x{shape[2]}"
            else:
                shape_text = f"{shape[0]}"
            
            # Create a more informative label
            full_label = f"{label}\nOutput: {shape_text}\nRF: {rf}x{rf}\nParams: {params:,}"
            
            # Draw the node
            plt.plot(x, y, 'o', markersize=20, color='lightgreen')
            plt.text(x, y, full_label, ha='center', va='center', fontsize=8)
    
    # Draw edges
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2], '-', color='gray')
    
    # Set axis limits with some padding
    plt.xlim(-0.5, max(x for x, y in pos.values()) + 0.5)
    plt.ylim(min(y for x, y in pos.values()) - 0.5, max(y for x, y in pos.values()) + 0.5)
    
    # Remove axis ticks and labels
    plt.axis('off')
    
    # Add title and description
    plt.title("Neural Network Architecture")
    
    return fig

def architecture_builder():
    st.title("CNN Architecture Builder")
    
    st.markdown("""
    # Interactive CNN Architecture Builder
    
    Design and visualize your own Convolutional Neural Network architecture. 
    Add layers, customize parameters, and see the network architecture visualization in real time.
    Generate ready-to-use code for PyTorch or TensorFlow.

    ## Features:
    - **Interactive Layer Addition**: Drag and drop layers to build your network
    - **Real-time Visualization**: See how your network architecture evolves
    - **Parameter Configuration**: Customize each layer's parameters
    - **Performance Estimation**: Get estimates for the model's size and computational requirements
    - **Code Generation**: Export your architecture as PyTorch or TensorFlow code
    """)
    
    # Framework selection
    framework = st.radio("Select Framework:", ["PyTorch", "TensorFlow"], horizontal=True)
    
    # Initialize session state for layers if not present
    if 'network_layers' not in st.session_state:
        st.session_state.network_layers = []
    
    # Input shape configuration
    st.subheader("Input Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        channels = st.slider("Input Channels:", 1, 3, 1)
    with col2:
        height = st.slider("Height:", 16, 256, 32)
    with col3:
        width = st.slider("Width:", 16, 256, 32)
    
    input_shape = (channels, height, width)
    
    # Display input shape
    st.markdown(f"**Input Shape**: {input_shape} (channels, height, width)")
    
    # Layer addition section
    st.subheader("Add Layers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        layer_type = st.selectbox("Layer Type:", list(LAYER_TYPES.keys()))
    
    # Get the parameters for the selected layer type
    layer_params = LAYER_TYPES[layer_type]["params"]
    
    # Parameter inputs based on layer type
    param_values = {}
    with col2:
        for param_name, param_config in layer_params.items():
            param_type = param_config["type"]
            
            if param_type == "int":
                param_values[param_name] = st.slider(
                    f"{param_name.capitalize()}:",
                    param_config["min"],
                    param_config["max"],
                    param_config["default"],
                    key=f"{layer_type}_{param_name}"
                )
            elif param_type == "float":
                param_values[param_name] = st.slider(
                    f"{param_name.capitalize()}:",
                    float(param_config["min"]),
                    float(param_config["max"]),
                    float(param_config["default"]),
                    step=0.01,
                    key=f"{layer_type}_{param_name}"
                )
            elif param_type == "select":
                param_values[param_name] = st.selectbox(
                    f"{param_name.capitalize()}:",
                    param_config["options"],
                    index=param_config["options"].index(param_config["default"]),
                    key=f"{layer_type}_{param_name}"
                )
    
    # Add layer button
    if st.button("Add Layer"):
        new_layer = {
            "type": layer_type,
            "params": param_values
        }
        st.session_state.network_layers.append(new_layer)
        st.success(f"Added {layer_type} layer")
    
    # Display and modify the current architecture
    if st.session_state.network_layers:
        st.subheader("Current Architecture")
        
        # Show layers in a table
        layer_data = []
        for i, layer in enumerate(st.session_state.network_layers):
            layer_info = {
                "Index": i+1,
                "Type": layer["type"],
                "Parameters": ", ".join([f"{k}={v}" for k, v in layer["params"].items()])
            }
            layer_data.append(layer_info)
        
        # Convert to a more readable format
        import pandas as pd
        layers_df = pd.DataFrame(layer_data)
        st.table(layers_df)
        
        # Option to remove a layer
        layer_to_remove = st.number_input("Remove Layer (enter layer index):", 
                                          min_value=1, 
                                          max_value=len(st.session_state.network_layers),
                                          value=len(st.session_state.network_layers))
        
        if st.button("Remove Layer"):
            if 1 <= layer_to_remove <= len(st.session_state.network_layers):
                removed_layer = st.session_state.network_layers.pop(layer_to_remove - 1)
                st.success(f"Removed {removed_layer['type']} layer at position {layer_to_remove}")
        
        # Calculate network properties
        output_shapes = calculate_output_shape(st.session_state.network_layers, input_shape)
        receptive_fields = calculate_receptive_field(st.session_state.network_layers, input_shape)
        param_counts, total_params = calculate_parameter_count(st.session_state.network_layers, input_shape)
        
        # Display the architecture visualization
        st.subheader("Network Visualization")
        fig = create_network_visualization(
            st.session_state.network_layers, 
            input_shape, 
            output_shapes, 
            receptive_fields, 
            param_counts
        )
        st.pyplot(fig)
        
        # Display network statistics
        st.subheader("Network Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        
        with col2:
            # Calculate the model size (rough approximation assuming 4 bytes per parameter)
            model_size_bytes = total_params * 4
            if model_size_bytes < 1024:
                model_size = f"{model_size_bytes} bytes"
            elif model_size_bytes < 1024 * 1024:
                model_size = f"{model_size_bytes / 1024:.2f} KB"
            else:
                model_size = f"{model_size_bytes / (1024 * 1024):.2f} MB"
            
            st.metric("Estimated Model Size", model_size)
        
        with col3:
            # Final output shape
            final_shape = output_shapes[-1]
            shape_str = "x".join(map(str, final_shape))
            st.metric("Output Shape", shape_str)
        
        # Code generation
        st.subheader("Generated Code")
        
        if framework == "PyTorch":
            code = generate_pytorch_code(st.session_state.network_layers, input_shape)
        else:  # TensorFlow
            code = generate_tensorflow_code(st.session_state.network_layers, input_shape)
        
        st.code(code, language="python")
        
        # Add a download button for the code
        if framework == "PyTorch":
            filename = "pytorch_model.py"
        else:
            filename = "tensorflow_model.py"
        
        st.download_button(
            label=f"Download {framework} Code",
            data=code,
            file_name=filename,
            mime="text/plain"
        )
        
        # Clear architecture button
        if st.button("Clear Architecture"):
            st.session_state.network_layers = []
            st.success("Architecture cleared")
    
    else:
        st.info("Start adding layers to build your CNN architecture")
    
    # Add educational content
    st.subheader("Understanding CNN Architecture Components")
    
    with st.expander("🧠 Convolutional Layers", expanded=False):
        st.markdown("""
        ### Convolutional Layers
        
        Convolutional layers are the core building blocks of CNN architectures. They apply filters to detect features in the input image.
        
        **Parameters:**
        - **Filters**: Number of feature detectors (kernels)
        - **Kernel Size**: Dimensions of the convolutional filter (typically 3x3 or 5x5)
        - **Stride**: Step size when sliding the filter (affects output spatial dimensions)
        - **Padding**: Adding pixels around the input (keeps spatial dimensions)
        
        **Function**: Extract spatial hierarchies of features (edges → textures → patterns → parts → objects)
        
        **Output size formula**:
        - With "valid" padding: `(input_size - kernel_size + 1) / stride`
        - With "same" padding: `input_size / stride`
        """)
    
    with st.expander("🏊 Pooling Layers", expanded=False):
        st.markdown("""
        ### Pooling Layers
        
        Pooling layers reduce the spatial dimensions (width, height) of the data, while preserving important features.
        
        **Types:**
        - **Max Pooling**: Takes the maximum value in each window
        - **Average Pooling**: Takes the average of all values in each window
        
        **Parameters:**
        - **Pool Size**: Dimensions of the pooling window (typically 2x2)
        - **Stride**: Step size when sliding the window (typically equal to pool size)
        
        **Benefits**:
        - Reduces computation for upper layers
        - Provides translational invariance
        - Controls overfitting by reducing parameters
        
        **Output size formula**: `(input_size - pool_size) / stride + 1`
        """)
    
    with st.expander("🔋 Activation Functions", expanded=False):
        st.markdown("""
        ### Activation Functions
        
        Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.
        
        **Common Types:**
        - **ReLU**: `f(x) = max(0, x)` - Fast, simple, and works well in most cases
        - **Sigmoid**: `f(x) = 1 / (1 + e^-x)` - Maps values to [0,1] range
        - **Tanh**: `f(x) = (e^x - e^-x) / (e^x + e^-x)` - Maps values to [-1,1] range
        - **LeakyReLU**: `f(x) = max(αx, x)` - Prevents "dying ReLU" problem
        
        **Key Considerations**:
        - ReLU is typically used in hidden layers
        - Sigmoid/Softmax commonly used for final layer in classification
        """)
    
    with st.expander("🧮 Other Important Layers", expanded=False):
        st.markdown("""
        ### Other Important Layers
        
        **Batch Normalization**:
        - Normalizes layer inputs for each mini-batch
        - Accelerates training and improves stability
        - Usually placed after Conv or FC layers, before activation
        
        **Dropout**:
        - Randomly deactivates a percentage of neurons during training
        - Prevents overfitting by forcing network redundancy
        - Typically used after fully connected layers
        
        **Flatten**:
        - Converts the multi-dimensional feature maps to a 1D vector
        - Required before fully connected layers
        
        **Fully Connected (Dense) Layers**:
        - Connects every neuron to every neuron in the next layer
        - Typically used at the end of the network for classification
        - Parameters: number of output units and activation function
        """)
    
    with st.expander("🏗️ Architecture Design Patterns", expanded=False):
        st.markdown("""
        ### Architecture Design Patterns
        
        **Traditional CNN Pattern**:
        ```
        [Input] → [Conv → Activation]+ → [Pooling]? → [Repeat] → [Flatten] → [FC → Activation]+ → [Output]
        ```
        
        **Modern Pattern (ResNet-like)**:
        ```
        [Input] → [[Conv → BatchNorm → Activation] × N + Skip Connection] × M → [Global Pooling] → [FC] → [Output]
        ```
        
        **Common Configurations**:
        - Gradually increase filter count (e.g., 32 → 64 → 128) as spatial dimensions decrease
        - Add pooling after 2-3 convolutional layers
        - Use dropout (0.25-0.5) before fully connected layers to prevent overfitting
        - End with fully connected layers that gradually decrease in size to the number of classes
        
        **Receptive Field**:
        - Each neuron in deeper layers "sees" a larger portion of the input
        - Important consideration for object detection and segmentation tasks
        """)

if __name__ == "__main__":
    architecture_builder()