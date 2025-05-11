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
from torchvision import transforms, models
import matplotlib.cm as cm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# Class names for MNIST dataset 
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# ImageNet classes (top 20 for simplicity in the UI)
IMAGENET_CLASSES_SUBSET = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
    'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house finch', 'junco', 'indigo bunting', 'American robin',
    'bulbul', 'jay', 'magpie', 'chickadee'
]

# Common image classification models
MODEL_OPTIONS = {
    "PyTorch": ["MNIST CNN", "ResNet-18 (ImageNet)"],
    "TensorFlow": ["MNIST CNN", "MobileNetV2 (ImageNet)"]
}

def create_mnist_cnn_pytorch():
    """Create a simple CNN model for MNIST classification"""
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def create_mnist_cnn_tensorflow():
    """Create a simple CNN model for MNIST classification in TensorFlow"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

def preprocess_image_pytorch(image, model_type):
    """Preprocess an image for PyTorch models"""
    if model_type == "MNIST CNN":
        # Convert to grayscale and resize to 28x28
        img = image.convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        
        # Add batch and channel dimensions
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return img_tensor, img_array
    
    elif model_type == "ResNet-18 (ImageNet)":
        # Standard ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        # Also prepare a normalized version for visualization
        img_array = np.array(image.resize((224, 224))) / 255.0
        
        return img_tensor, img_array

def preprocess_image_tensorflow(image, model_type):
    """Preprocess an image for TensorFlow models"""
    if model_type == "MNIST CNN":
        # Convert to grayscale and resize to 28x28
        img = image.convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        
        # Add batch and channel dimensions
        img_tensor = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)
        return img_tensor, img_array
    
    elif model_type == "MobileNetV2 (ImageNet)":
        # Resize to 224x224
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        # Preprocess with TF's preprocess_input for MobileNetV2
        img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(img_array, axis=0)
        )
        
        return img_tensor, img_array

def get_class_activation_map(model, img_tensor, img_array, framework, model_type):
    """Generate a CAM (Class Activation Map) for the model's prediction"""
    if framework == "PyTorch":
        if model_type == "MNIST CNN":
            # For custom MNIST model, use a simple approach to simulate CAM
            # Extract activations from the last convolutional layer
            activations = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach()
                return hook
            
            # Register a hook for the second conv layer
            for name, module in model.named_children():
                if isinstance(module, nn.Conv2d) and name == '2':  # second conv layer
                    hook = module.register_forward_hook(get_activation('2'))
                    break
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
                prediction = output.argmax(1).item()
            
            # Remove hook
            hook.remove()
            
            # Generate a simple "heatmap" based on feature activations
            feature_maps = activations['2'].squeeze().cpu().numpy()
            
            # Use mean across channels for simplicity
            cam = np.mean(feature_maps, axis=0)
            
            # Resize to match input
            cam = cv2.resize(cam, (28, 28))
            
            # Normalize to 0-1 range
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam, prediction
        
        elif model_type == "ResNet-18 (ImageNet)":
            # Use Grad-CAM approach
            # Register hook for gradients
            gradients = []
            activations = []
            
            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])
                return None
            
            def forward_hook(module, input, output):
                activations.append(output)
                return None
            
            # Get the last convolutional layer in ResNet
            last_conv_layer = None
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv_layer = module
            
            # Register hooks
            forward_handle = last_conv_layer.register_forward_hook(forward_hook)
            backward_handle = last_conv_layer.register_backward_hook(backward_hook)
            
            # Forward pass
            model.eval()
            output = model(img_tensor)
            prediction = output.argmax(1).item()
            
            # Backward pass for the predicted class
            model.zero_grad()
            class_idx = output.argmax(1)
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1
            output.backward(gradient=one_hot)
            
            # Clean up hooks
            forward_handle.remove()
            backward_handle.remove()
            
            # Compute Grad-CAM
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations[0]).sum(dim=1, keepdim=True)
            cam = F.relu(cam)  # Apply ReLU to highlight positive contributions
            
            # Normalize and resize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (224, 224))
            
            return cam, prediction
    
    elif framework == "TensorFlow":
        if model_type == "MNIST CNN":
            # For custom MNIST model, create a simpler visualization
            # Get the output of the second-to-last convolutional layer
            feature_extractor = tf.keras.Model(
                inputs=model.inputs,
                outputs=model.get_layer(index=-3).output  # Last conv layer before flatten
            )
            
            # Get the feature maps
            feature_maps = feature_extractor(img_tensor)
            feature_maps = feature_maps.numpy()[0]  # Remove batch dimension
            
            # Use mean across channels
            cam = np.mean(feature_maps, axis=-1)
            
            # Resize to input size
            cam = cv2.resize(cam, (28, 28))
            
            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Get prediction
            output = model(img_tensor)
            prediction = np.argmax(output[0])
            
            return cam, prediction
        
        elif model_type == "MobileNetV2 (ImageNet)":
            # Use TF's GradientTape for Grad-CAM
            last_conv_layer = None
            
            # Find the last convolutional layer
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
            
            # Create a model that outputs both the final prediction and the activations
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs], 
                outputs=[model.get_layer(last_conv_layer).output, model.output]
            )
            
            # Compute gradient using GradientTape
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_tensor)
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
            
            # Gradient of the predicted class with respect to the last conv layer
            grads = tape.gradient(class_channel, conv_output)
            
            # Global average pooling of the gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps with gradients
            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
            
            # Apply ReLU and normalize
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
            heatmap = heatmap.numpy()
            
            # Resize to original image size
            cam = cv2.resize(heatmap, (224, 224))
            
            # Get prediction
            prediction = pred_index.numpy()
            
            return cam, prediction
    
    # Default return if no valid combination is found
    return np.zeros_like(img_array), 0

def generate_activation_heatmap(img, cam):
    """Overlay a heatmap on an image to visualize class activation"""
    # Convert CAM to heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image to RGB if grayscale
    if len(img.shape) == 2:
        img_rgb = np.stack([img]*3, axis=-1)
    else:
        img_rgb = img
    
    # Ensure image is in the right format (0-255 range)
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape[:2] != img_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # Blend image with heatmap
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    
    return superimposed_img

def get_feature_visualizations(model, img_tensor, framework, model_type):
    """Extract feature maps from different layers of the model"""
    feature_maps = []
    
    if framework == "PyTorch":
        if model_type == "MNIST CNN":
            # Extract activations from each conv layer
            activations = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach()
                return hook
            
            # Register hooks for convolutional layers
            handles = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    handles.append(module.register_forward_hook(get_activation(name)))
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(img_tensor)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            # Format activations
            for name, activation in activations.items():
                act = activation.squeeze().cpu().numpy()
                if len(act.shape) == 3:  # Conv layer with multiple channels
                    feature_maps.append({
                        "name": f"Conv Layer {name}",
                        "maps": act,
                        "shape": act.shape
                    })
        
        elif model_type == "ResNet-18 (ImageNet)":
            # Extract activations from selected ResNet layers
            activations = {}
            
            def get_activation(name):
                def hook(module, input, output):
                    activations[name] = output.detach()
                return hook
            
            # Register hooks for key layers
            handles = []
            layer_names = ["layer1", "layer2", "layer3", "layer4"]
            for name, module in model.named_children():
                if name in layer_names:
                    handles.append(module.register_forward_hook(get_activation(name)))
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(img_tensor)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            # Format activations
            for name, activation in activations.items():
                act = activation.squeeze().cpu().numpy()
                if len(act.shape) == 3:  # Conv layer with multiple channels
                    # For ResNet, pick a few representative channels
                    n_channels = min(8, act.shape[0])
                    feature_maps.append({
                        "name": f"ResNet {name}",
                        "maps": act[:n_channels],
                        "shape": act.shape
                    })
    
    elif framework == "TensorFlow":
        if model_type == "MNIST CNN":
            # Create a model that outputs activations from all layers
            layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
            activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
            
            # Get activations
            activations = activation_model.predict(img_tensor)
            
            # Format activations
            for i, activation in enumerate(activations):
                act = activation[0]  # Remove batch dimension
                feature_maps.append({
                    "name": f"Conv Layer {i+1}",
                    "maps": act,
                    "shape": act.shape
                })
        
        elif model_type == "MobileNetV2 (ImageNet)":
            # Extract activations from selected MobileNet layers
            layer_names = ["block_1_expand", "block_3_expand", "block_6_expand", "block_13_expand"]
            layer_outputs = []
            
            for name in layer_names:
                try:
                    # Try to find the layer by name
                    layer = model.get_layer(name)
                    layer_outputs.append(layer.output)
                except:
                    # Skip if layer not found
                    continue
            
            # Create a model that outputs these activations
            if layer_outputs:
                activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                
                # Get activations
                activations = activation_model.predict(img_tensor)
                
                # Format activations
                for i, activation in enumerate(activations):
                    act = activation[0]  # Remove batch dimension
                    # For MobileNet, pick a few representative channels
                    n_channels = min(8, act.shape[-1])
                    feature_maps.append({
                        "name": f"MobileNet Block {i+1}",
                        "maps": np.transpose(act[:, :, :n_channels], (2, 0, 1)),  # Reorder to match PyTorch format
                        "shape": act.shape
                    })
    
    return feature_maps

def visualize_feature_maps(feature_maps):
    """Visualize feature maps from different layers"""
    for layer_data in feature_maps:
        st.subheader(f"{layer_data['name']} (Shape: {layer_data['shape']})")
        
        maps = layer_data["maps"]
        n_channels = maps.shape[0]
        
        # Create a grid of feature maps
        n_cols = min(4, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i in range(n_channels):
            if i < len(axes):
                im = axes[i].imshow(maps[i], cmap='viridis')
                plt.colorbar(im, ax=axes[i])
                axes[i].set_title(f"Channel {i+1}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Also show 3D visualization for one channel
        if n_channels > 0:
            feature_map = maps[0]
            
            # Create 3D visualization with Plotly
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
                title=f"3D View of Feature Map (Channel 1)",
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

def image_classification_playground():
    st.title("Image Classification Playground")
    
    st.markdown("""
    # Interactive Image Classification
    
    This playground lets you explore how different CNN models classify images. 
    Upload your own images or use our sample images to see the model's predictions, 
    along with visualizations of the model's internal representations.
    
    ## Features:
    - **Image Classification**: See predictions from different models
    - **Activation Heatmaps**: Visualize which regions of the image the model focuses on
    - **Feature Maps**: Explore the internal representations learned by the network
    - **Interactive Visualizations**: Examine model behavior through intuitive displays
    """)
    
    # Framework selection
    framework = st.radio("Select Framework:", ["PyTorch", "TensorFlow"], horizontal=True)
    
    # Model selection based on framework
    model_type = st.selectbox("Select Model:", MODEL_OPTIONS[framework])
    
    # Load the selected model
    model = None
    class_names = []
    
    with st.spinner("Loading model..."):
        if framework == "PyTorch":
            if model_type == "MNIST CNN":
                model = create_mnist_cnn_pytorch()
                # In a real app, we would load pre-trained weights here
                # For demo purposes, we'll use the untrained model
                class_names = MNIST_CLASSES
            
            elif model_type == "ResNet-18 (ImageNet)":
                model = models.resnet18(pretrained=True)
                model.eval()
                class_names = IMAGENET_CLASSES_SUBSET  # Using our subset for UI simplicity
        
        elif framework == "TensorFlow":
            if model_type == "MNIST CNN":
                model = create_mnist_cnn_tensorflow()
                # For demo purposes, we'll use the untrained model
                class_names = MNIST_CLASSES
            
            elif model_type == "MobileNetV2 (ImageNet)":
                model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=True,
                    weights='imagenet'
                )
                class_names = IMAGENET_CLASSES_SUBSET  # Using our subset for UI simplicity
    
    # Image input
    st.subheader("Image Input")
    
    # Option to use sample image or upload own
    input_option = st.radio("Select input method:", 
                          ["Upload Image", "Use Sample Image"], 
                          horizontal=True)
    
    image = None
    
    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    else:  # Use Sample Image
        sample_type = "Digit" if "MNIST" in model_type else "Object"
        
        if sample_type == "Digit":
            # Generate some digit-like images for MNIST models
            digits = [
                np.zeros((28, 28)),  # 0
                np.zeros((28, 28)),  # 1
                np.zeros((28, 28)),  # 2
                np.zeros((28, 28)),  # 3
                np.zeros((28, 28))   # 4
            ]
            
            # Create simple patterns for each digit
            # 0: Circle
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - 14)**2 + (j - 14)**2)
                    if 8 < dist < 12:
                        digits[0][i, j] = 1.0
            
            # 1: Vertical line
            digits[1][:, 14:16] = 1.0
            
            # 2: Rough shape of 2
            digits[2][5:8, 5:20] = 1.0
            digits[2][8:15, 17:20] = 1.0
            digits[2][15:18, 5:20] = 1.0
            digits[2][18:22, 5:8] = 1.0
            digits[2][22:25, 5:20] = 1.0
            
            # 3: Rough shape of 3
            digits[3][5:8, 5:20] = 1.0
            digits[3][8:12, 17:20] = 1.0
            digits[3][12:15, 10:20] = 1.0
            digits[3][15:18, 17:20] = 1.0
            digits[3][18:25, 5:20] = 1.0
            
            # 4: Rough shape of 4
            digits[4][5:18, 17:20] = 1.0
            digits[4][15:18, 5:25] = 1.0
            digits[4][18:25, 17:20] = 1.0
            
            sample_options = {f"Digit {i}": digits[i] for i in range(5)}
            
            selected_sample = st.selectbox("Select a sample digit:", list(sample_options.keys()))
            digit_img = sample_options[selected_sample]
            
            # Convert the array to a PIL Image
            image = Image.fromarray((digit_img * 255).astype(np.uint8))
            
            # Display the sample
            st.image(image, caption=f"Sample {selected_sample}", use_container_width=True, clamp=True)
        
        else:  # Object samples for ImageNet models
            # Generate a few colored shapes to serve as sample images
            shapes = [
                ("Red Square", np.zeros((224, 224, 3))),
                ("Green Circle", np.zeros((224, 224, 3))),
                ("Blue Triangle", np.zeros((224, 224, 3))),
                ("Yellow Stars", np.zeros((224, 224, 3))),
                ("Gradient", np.zeros((224, 224, 3)))
            ]
            
            # Create simple shapes with different colors
            # Red Square
            shapes[0][1][50:174, 50:174, 0] = 1.0
            
            # Green Circle
            center = (112, 112)
            radius = 80
            for i in range(224):
                for j in range(224):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if dist < radius:
                        shapes[1][1][i, j, 1] = 1.0
            
            # Blue Triangle
            triangle = np.array([
                [112, 40],  # top
                [40, 180],  # bottom left
                [184, 180]  # bottom right
            ])
            img = np.zeros((224, 224, 3))
            cv2.fillPoly(img, [triangle], (0, 0, 1))
            shapes[2][1] = img
            
            # Yellow Stars (multiple small shapes)
            img = np.zeros((224, 224, 3))
            for _ in range(20):
                x = np.random.randint(20, 204)
                y = np.random.randint(20, 204)
                size = np.random.randint(10, 20)
                pts = np.array([
                    [x, y - size],
                    [x + size//2, y - size//4],
                    [x + size, y],
                    [x + size//2, y + size//4],
                    [x, y + size],
                    [x - size//2, y + size//4],
                    [x - size, y],
                    [x - size//2, y - size//4]
                ])
                cv2.fillPoly(img, [pts], (1, 1, 0))
            shapes[3][1] = img
            
            # Gradient
            for i in range(224):
                for j in range(224):
                    shapes[4][1][i, j, 0] = i / 224
                    shapes[4][1][i, j, 1] = j / 224
                    shapes[4][1][i, j, 2] = (i + j) / (2 * 224)
            
            sample_options = {name: img for name, img in shapes}
            
            selected_sample = st.selectbox("Select a sample object:", list(sample_options.keys()))
            object_img = sample_options[selected_sample]
            
            # Convert the array to a PIL Image
            image = Image.fromarray((object_img * 255).astype(np.uint8))
            
            # Display the sample
            st.image(image, caption=f"Sample {selected_sample}", use_container_width=True)
    
    # Proceed with classification if an image is available
    if image is not None:
        st.subheader("Classification Results")
        
        # Process the image and get predictions
        if framework == "PyTorch":
            img_tensor, img_array = preprocess_image_pytorch(image, model_type)
            
            # Get model prediction
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)[0].numpy()
                
                # Get top prediction
                prediction = output.argmax(1).item()
        
        elif framework == "TensorFlow":
            img_tensor, img_array = preprocess_image_tensorflow(image, model_type)
            
            # Get model prediction
            output = model.predict(img_tensor)
            
            # Apply softmax to get probabilities
            if model_type == "MNIST CNN":
                probabilities = tf.nn.softmax(output[0]).numpy()
            else:
                probabilities = output[0]
            
            # Get top prediction
            prediction = np.argmax(probabilities)
        
        # Display prediction
        if model_type in ["MNIST CNN"]:
            predicted_class = class_names[prediction]
            st.markdown(f"### Predicted Digit: **{predicted_class}**")
        else:
            # For ImageNet models, use the subset of classes
            if prediction < len(class_names):
                predicted_class = class_names[prediction]
            else:
                predicted_class = f"Class #{prediction}"
            st.markdown(f"### Predicted Class: **{predicted_class}**")
        
        # Plot probability distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Limit to showing only a reasonable number of classes
        num_classes_to_show = min(20, len(probabilities))
        indices = np.argsort(probabilities)[-num_classes_to_show:][::-1]
        
        # Get class names and probabilities for display
        plot_classes = [class_names[i] if i < len(class_names) else f"Class {i}" for i in indices]
        plot_probs = probabilities[indices]
        
        # Create horizontal bar chart
        bars = ax.barh(plot_classes, plot_probs, color='skyblue')
        ax.set_title("Class Probabilities")
        ax.set_xlabel("Probability")
        ax.grid(True, alpha=0.3)
        
        # Highlight the predicted class
        for i, bar in enumerate(bars):
            if plot_classes[i] == predicted_class:
                bar.set_color('green')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualization section
        st.subheader("Visualization of Model Focus")
        
        # Generate class activation map
        cam, _ = get_class_activation_map(model, img_tensor, img_array, framework, model_type)
        
        # Apply heatmap
        heatmap_image = generate_activation_heatmap(img_array, cam)
        
        # Display original and heatmap side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("**Class Activation Map**")
            st.image(heatmap_image, use_container_width=True)
            st.markdown("""
            The highlighted regions show areas the model focuses on when making its prediction.
            Warmer colors (red/yellow) indicate higher importance to the classification decision.
            """)
        
        # Feature map visualization
        st.subheader("Layer Feature Maps")
        
        st.markdown("""
        These visualizations show how different layers of the neural network "see" the input image. 
        Early layers detect simple features like edges and textures, while deeper layers recognize more complex patterns.
        
        The 3D representations show the activation strength across the feature map - peaks represent strong feature detections.
        """)
        
        # Get feature maps from different layers
        feature_maps = get_feature_visualizations(model, img_tensor, framework, model_type)
        
        # Visualize them
        visualize_feature_maps(feature_maps)
        
        # Educational explanation
        st.subheader("Understanding the Classification Process")
        
        st.markdown("""
        ## How Neural Networks Classify Images
        
        1. **Feature Extraction**: Convolutional layers detect progressively more complex features
           - Early layers: Edges, textures, and simple shapes
           - Middle layers: Parts and combinations of features
           - Deep layers: Complete objects and abstract concepts
        
        2. **Spatial Understanding**: Pooling layers reduce dimensions while preserving important information
        
        3. **Decision Making**: Fully connected layers combine features to make the final classification
        
        The visualizations above reveal what the network "sees" at each stage of this process.
        """)
        
        # Add interactive explanation based on the model type
        if "MNIST" in model_type:
            st.markdown("""
            ### Digit Recognition
            
            For digit recognition, the model learns:
            - Distinctive shapes and curves of each digit
            - Relative positions of strokes
            - Common variations in handwriting styles
            
            The MNIST dataset provides a controlled environment for studying image classification basics.
            """)
        else:
            st.markdown("""
            ### Object Recognition
            
            For object recognition, the model learns:
            - Hierarchical composition of features (edges → textures → parts → objects)
            - Invariance to position, lighting, and viewpoint
            - Distinguishing characteristics between similar classes
            
            ImageNet-trained models have learned from millions of diverse images, allowing them to recognize thousands of different object categories.
            """)

if __name__ == "__main__":
    image_classification_playground()