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
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import cm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

def create_simple_cnn():
    """Create a simple CNN model for visualization"""
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 7 * 7, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    return model

def create_simple_tf_model():
    """Create a simple TensorFlow CNN model for visualization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

def generate_dummy_mnist_data(n_samples=100):
    """Generate a small dummy MNIST-like dataset"""
    # Generate random images (28x28)
    x = np.random.rand(n_samples, 28, 28) * 0.3
    
    # Create some random "digits" as circular blobs
    for i in range(n_samples):
        # Random digit class (0-9)
        digit = np.random.randint(0, 10)
        
        # Create a circular blob for that digit
        center_x = 14 + np.random.randint(-5, 5)
        center_y = 14 + np.random.randint(-5, 5)
        radius = 6 + np.random.randint(-2, 2)
        
        for x_coord in range(28):
            for y_coord in range(28):
                # Calculate distance from center
                dist = np.sqrt((x_coord - center_x)**2 + (y_coord - center_y)**2)
                if dist < radius:
                    # Add intensity to represent the digit
                    x[i, x_coord, y_coord] = 0.7 + np.random.rand() * 0.3
        
        # Add some variations based on digit
        if digit % 2 == 0:  # Even digits
            # Add a horizontal line
            line_y = 10 + (digit // 2) * 2
            x[i, :, line_y:line_y+2] = x[i, :, line_y:line_y+2] + 0.3
            x[i, x[i] > 1.0] = 1.0  # Clip values
        else:  # Odd digits
            # Add a vertical line
            line_x = 10 + (digit // 2) * 2
            x[i, line_x:line_x+2, :] = x[i, line_x:line_x+2, :] + 0.3
            x[i, x[i] > 1.0] = 1.0  # Clip values
    
    # Create corresponding labels
    y = np.random.randint(0, 10, size=n_samples)
    
    # Reshape x for CNN
    x_reshaped = x.reshape(n_samples, 28, 28, 1)
    
    return x_reshaped, y

def get_layer_activations(model, x, framework="pytorch"):
    """Extract activations from each layer of the model"""
    activations = {}
    
    if framework == "pytorch":
        # Convert input to PyTorch format
        if len(x.shape) == 4:  # (batch, h, w, channels)
            x_torch = torch.tensor(np.transpose(x, (0, 3, 1, 2)), dtype=torch.float32)
        else:  # Single image
            x_torch = torch.tensor(np.transpose(x, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        
        # Register hooks to capture activations
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().numpy()
            return hook
        
        # Register hooks for each layer
        handles = []
        for name, module in model.named_children():
            handles.append(module.register_forward_hook(get_activation(name)))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(x_torch)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
    elif framework == "tensorflow":
        # Ensure input is in TensorFlow format
        if len(x.shape) == 3:  # Single image
            x_tf = np.expand_dims(x, axis=0)
        else:
            x_tf = x
        
        # Create separate models for each layer
        layer_outputs = []
        for i, layer in enumerate(model.layers):
            layer_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
            layer_outputs.append(layer_model.predict(x_tf))
        
        # Store activations
        for i, output in enumerate(layer_outputs):
            layer_name = model.layers[i].name
            activations[layer_name] = output
    
    return activations

def visualize_training_progress():
    st.title("Layer-by-Layer Training Visualization")
    
    st.markdown("""
    # Interactive CNN Training Visualization
    
    This section allows you to visualize how a Convolutional Neural Network learns and evolves during training.
    See how each layer's activations change as the model learns to recognize patterns.
    
    ## Features:
    - **Layer Activations**: Visualize how different layers represent the input data
    - **Training Evolution**: See how representations change during training
    - **3D Visualizations**: Explore feature maps in three dimensions
    - **Comparison**: Compare PyTorch and TensorFlow implementations
    """)
    
    # Framework selection
    framework = st.radio("Select Framework:", ["PyTorch", "TensorFlow"], horizontal=True)
    
    # Generate sample data
    if 'dummy_data' not in st.session_state:
        st.session_state.dummy_data = generate_dummy_mnist_data(n_samples=200)
    
    x_train, y_train = st.session_state.dummy_data
    
    # Create model
    if framework == "PyTorch":
        if 'pytorch_model' not in st.session_state:
            st.session_state.pytorch_model = create_simple_cnn()
        model = st.session_state.pytorch_model
    else:  # TensorFlow
        if 'tf_model' not in st.session_state:
            st.session_state.tf_model = create_simple_tf_model()
        model = st.session_state.tf_model
    
    # Show model architecture
    st.subheader("Model Architecture")
    
    if framework == "PyTorch":
        # Display PyTorch model architecture
        st.code(str(model), language="python")
    else:
        # Display TensorFlow model summary
        string_io = io.StringIO()
        model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        model_summary = string_io.getvalue()
        st.code(model_summary, language="")
    
    # Training simulation options
    st.subheader("Training Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs:", 1, 10, 5)
        batch_size = st.slider("Batch Size:", 8, 64, 32)
    
    with col2:
        learning_rate = st.slider("Learning Rate:", 0.001, 0.1, 0.01, step=0.001)
        show_layer = st.selectbox("Layer to Visualize:", 
                                 ["Conv2d_1", "ReLU_1", "MaxPool2d_1", 
                                  "Conv2d_2", "ReLU_2", "MaxPool2d_2", 
                                  "Linear_1", "ReLU_3", "Linear_2"])
    
    # Sample images to track throughout training
    num_samples = 5
    
    # Select a few sample images for visualization
    if 'viz_samples' not in st.session_state:
        random_indices = np.random.choice(len(x_train), num_samples, replace=False)
        st.session_state.viz_samples = x_train[random_indices], y_train[random_indices]
    
    viz_x, viz_y = st.session_state.viz_samples
    
    # Display sample images
    st.subheader("Sample Images for Visualization")
    
    cols = st.columns(num_samples)
    for i, col in enumerate(cols):
        with col:
            st.image(viz_x[i, :, :, 0], caption=f"Label: {viz_y[i]}", use_container_width=True, clamp=True)
    
    # Training simulation
    if st.button("Run Training Simulation"):
        if framework == "PyTorch":
            # Set up PyTorch model and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
            # Convert data to PyTorch format
            x_torch = torch.tensor(np.transpose(x_train, (0, 3, 1, 2)), dtype=torch.float32)
            y_torch = torch.tensor(y_train, dtype=torch.long)
            
            # Create DataLoader
            dataset = torch.utils.data.TensorDataset(x_torch, y_torch)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Prepare viz samples
            viz_x_torch = torch.tensor(np.transpose(viz_x, (0, 3, 1, 2)), dtype=torch.float32)
            
            # Create progress bar and visualization containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_plot = st.empty()
            activation_plot = st.empty()
            
            # Track loss history
            losses = []
            
            # Run training
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                for x_batch, y_batch in dataloader:
                    # Forward pass
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Update progress
                progress_bar.progress((epoch + 1) / epochs)
                avg_loss = epoch_loss / batch_count
                losses.append(avg_loss)
                status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
                # Plot loss
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(1, epoch+2), losses, 'b-')
                ax.set_title('Training Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                loss_plot.pyplot(fig)
                plt.close(fig)
                
                # Get activations for visualization
                activations = get_layer_activations(model, viz_x, framework="pytorch")
                
                # Visualize activations for the selected layer
                visualize_activations(activations, framework="pytorch")
                
                # Small delay for better visualization
                time.sleep(0.5)
        
        else:  # TensorFlow
            # Set up TensorFlow model and optimizer
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            # Create progress bar and visualization containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_plot = st.empty()
            activation_plot = st.empty()
            
            # Track loss history
            losses = []
            
            # Run training
            for epoch in range(epochs):
                # Fit for one epoch
                history = model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    epochs=1,
                    verbose=0
                )
                
                # Update progress
                progress_bar.progress((epoch + 1) / epochs)
                epoch_loss = history.history['loss'][0]
                losses.append(epoch_loss)
                status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                
                # Plot loss
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(1, epoch+2), losses, 'b-')
                ax.set_title('Training Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                loss_plot.pyplot(fig)
                plt.close(fig)
                
                # Get activations for visualization
                activations = get_layer_activations(model, viz_x, framework="tensorflow")
                
                # Visualize activations
                visualize_activations(activations, framework="tensorflow")
                
                # Small delay for better visualization
                time.sleep(0.5)
        
        st.success("Training simulation complete!")
    
    # Activation visualization section
    st.subheader("Layer Activation Visualization")
    
    st.markdown("""
    This section allows you to explore how different layers of the CNN represent input data.
    Select a layer and see how it transforms the input images.
    """)
    
    # Get one sample for demonstration
    sample_x = viz_x[0:1]
    
    # Get activations
    activations = get_layer_activations(model, sample_x, framework=framework.lower())
    
    # Visualize them
    visualize_activations(activations, framework=framework.lower())
    
    # Additional explanation
    st.subheader("Understanding Layer Activations")
    
    st.markdown("""
    ### How to Interpret These Visualizations:
    
    1. **Early Convolutional Layers (Conv1)**: 
       - Detect basic features like edges and textures
       - Activations look like edge detection filters
    
    2. **Middle Layers (Conv2)**:
       - Combine basic features into more complex patterns
       - Activations represent higher-level features
    
    3. **Later Layers (Dense/Linear)**:
       - Extract abstract representations relevant to classification
       - Harder to interpret visually, but represent class-related features
    
    ### The Training Process:
    
    As training progresses, you should notice:
    - Activations become more defined and specialized
    - Feature detectors evolve to better represent relevant patterns
    - The model gradually focuses on the most discriminative features
    
    Try running the training simulation multiple times to see how random initialization affects learning!
    """)

def visualize_activations(activations, framework):
    """Visualize activations from different layers"""
    # Get layer names
    layer_names = list(activations.keys())
    
    # Select up to 5 layers to display
    show_layers = layer_names[:min(5, len(layer_names))]
    
    # Create tabs for different visualization types
    tab1, tab2 = st.tabs(["2D Visualizations", "3D Feature Maps"])
    
    with tab1:
        # 2D visualizations of activations
        for layer_name in show_layers:
            activation = activations[layer_name]
            
            # Get the activation shape
            if framework == "pytorch":
                # PyTorch format: (batch, channels, height, width)
                if len(activation.shape) == 4:
                    batch_size, channels, height, width = activation.shape
                    st.markdown(f"**Layer {layer_name}** - Shape: {activation.shape}")
                    
                    # Show a few channels
                    num_channels = min(8, channels)
                    
                    # Create a grid of plots
                    fig, axs = plt.subplots(1, num_channels, figsize=(15, 3))
                    
                    # Handle the case of a single channel
                    if num_channels == 1:
                        axs = [axs]
                    
                    for i in range(num_channels):
                        im = axs[i].imshow(activation[0, i], cmap='viridis')
                        plt.colorbar(im, ax=axs[i])
                        axs[i].set_title(f"Channel {i+1}")
                        axs[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                else:  # Dense layer
                    st.markdown(f"**Layer {layer_name}** - Shape: {activation.shape}")
                    
                    # Plot activation values as a bar chart
                    fig, ax = plt.subplots(figsize=(12, 3))
                    features = min(32, activation.shape[1])  # Show at most 32 features
                    ax.bar(range(features), activation[0, :features])
                    ax.set_title(f"Neuron Activations for {layer_name}")
                    ax.set_xlabel("Neuron")
                    ax.set_ylabel("Activation")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            else:  # TensorFlow
                # TensorFlow format: (batch, height, width, channels)
                if len(activation.shape) == 4:
                    batch_size, height, width, channels = activation.shape
                    st.markdown(f"**Layer {layer_name}** - Shape: {activation.shape}")
                    
                    # Show a few channels
                    num_channels = min(8, channels)
                    
                    # Create a grid of plots
                    fig, axs = plt.subplots(1, num_channels, figsize=(15, 3))
                    
                    # Handle the case of a single channel
                    if num_channels == 1:
                        axs = [axs]
                    
                    for i in range(num_channels):
                        im = axs[i].imshow(activation[0, :, :, i], cmap='viridis')
                        plt.colorbar(im, ax=axs[i])
                        axs[i].set_title(f"Channel {i+1}")
                        axs[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                else:  # Dense layer
                    st.markdown(f"**Layer {layer_name}** - Shape: {activation.shape}")
                    
                    # Plot activation values as a bar chart
                    fig, ax = plt.subplots(figsize=(12, 3))
                    features = min(32, activation.shape[1])  # Show at most 32 features
                    ax.bar(range(features), activation[0, :features])
                    ax.set_title(f"Neuron Activations for {layer_name}")
                    ax.set_xlabel("Neuron")
                    ax.set_ylabel("Activation")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
    
    with tab2:
        # 3D visualizations of feature maps
        # Show only convolutional layers
        for layer_name in show_layers:
            activation = activations[layer_name]
            
            # Only create 3D visualization for convolutional layers
            if len(activation.shape) == 4:
                # Get dimensions based on framework
                if framework == "pytorch":
                    # PyTorch format: (batch, channels, height, width)
                    batch_size, channels, height, width = activation.shape
                    
                    # Select one channel to visualize in 3D
                    channel_idx = 0
                    feature_map = activation[0, channel_idx]
                
                else:  # TensorFlow
                    # TensorFlow format: (batch, height, width, channels)
                    batch_size, height, width, channels = activation.shape
                    
                    # Select one channel to visualize in 3D
                    channel_idx = 0
                    feature_map = activation[0, :, :, channel_idx]
                
                # Create 3D visualization with Plotly
                st.markdown(f"**3D Feature Map for Layer {layer_name}** (Channel {channel_idx+1})")
                
                # Create meshgrid for 3D plot
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
                
                # Add explanation of the 3D visualization
                st.markdown("""
                In this 3D visualization:
                - The x and y axes represent the spatial dimensions of the feature map
                - The z axis (height) represents the activation strength at each position
                - Peaks indicate strong activations (the feature detector found a match)
                - Valleys indicate weak or no activations
                
                You can rotate, zoom, and pan to explore the feature map in detail.
                """)

if __name__ == "__main__":
    visualize_training_progress()