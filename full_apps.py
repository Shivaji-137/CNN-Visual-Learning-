"""
Written by Shivaji Chaulagain.
This is under the MIT License.
If you use this code, please give credit to the author.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
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
import cnn_scratch
import cnn_tensorflow
import cnn_pytorch
import utils
import visualizations
import complete_cnn_copy

# Set page configuration
st.set_page_config(
    page_title="Complete CNN Visualization",
    page_icon="🧠",
    layout="wide"
)

st.title("Convolutional Neural Networks (CNNs) - Educational App")
st.markdown("""
This application provides an interactive exploration of Convolutional Neural Networks (CNNs),
implemented both from scratch using NumPy and with popular deep learning libraries 
(TensorFlow and PyTorch). Compare the implementations, visualize the CNN architecture,
and understand how CNNs work through interactive demonstrations.
""")

# Sidebar for navigation and parameters
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a section",
    ["Introduction", "CNN from Scratch", "Process Sample Images", "Upload Your Own Image", "Custom Filter Builder", "CNN with Libraries", "Comparison", "Training Visualization"]
)

st.sidebar.title("Filter Parameters")
filter_size = st.sidebar.slider("Filter Size", 3, 7, 3, 2)
all_filters = list(filters.COMMON_FILTERS.keys())

complete_cnn_copy.appmode(app_mode).call_app()

# Initialize with defaults to avoid unbound variable errors
filters_to_show = ["Edge Detection (Horizontal)", "Edge Detection (Vertical)", "Sobel (Horizontal)", "Sobel (Vertical)"]



# Common parameters in sidebar
st.sidebar.title("CNN Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, 16)
num_epochs = st.sidebar.slider("Number of Epochs", 1, 10, 3, 1)
conv_filters = st.sidebar.slider("Number of Convolutional Filters", 4, 32, 16, 4)


@st.cache_data
def load_mnist_data():
    digits = datasets.load_digits()
    # Normalize data
    X = digits.data / 16.0
    y = digits.target
    
    # Reshape for CNN (assuming 8x8 images for digits dataset)
    X = X.reshape(-1, 8, 8, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, digits

def load_olivetti_data():
    # Load Olivetti dataset
    olivetti = datasets.fetch_olivetti_faces()
    X = olivetti.data
    y = olivetti.target
    
    # Normalize data
    X = X / 16.0
    
    # Reshape for CNN (assuming 64x64 images for Olivetti dataset)
    X = X.reshape(-1, 64, 64, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, olivetti

# CNN from Scratch section
if app_mode == "CNN from Scratch":
    st.header("CNN Implementation from Scratch (using NumPy)")
    
    st.markdown("""
    This section demonstrates a CNN implemented using only NumPy for matrix operations.
    We'll visualize each component and step through the forward and backward propagation processes.
    """)
    
    # Load the dataset
    X_train, X_test, y_train, y_test, digits = load_mnist_data()
    
    # Display sample images
    st.subheader("Sample Images from Dataset")
    
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            img = X_train[i].reshape(8, 8)
            st.image(img, caption=f"Label: {y_train[i]}", use_container_width=True)
    
    # CNN architecture from scratch
    st.subheader("CNN Architecture (from scratch)")
    
    # Initialize CNN model
    scratch_model = cnn_scratch.CNN(
        input_shape=(8, 8, 1),
        conv_filters=conv_filters,
        learning_rate=learning_rate
    )
    
    # Display CNN architecture
    visualizations.display_scratch_cnn_architecture(scratch_model)
    
    # Convolution step by step
    st.subheader("Convolution Layer (from scratch)")
    selected_image_idx = st.slider("Select image", 0, len(X_test)-1, 0)
    selected_image = X_test[selected_image_idx]
    
    # Visualize the convolution operation
    visualizations.visualize_convolution_scratch(scratch_model, selected_image)
    
    # Pooling step by step
    st.subheader("Pooling Layer (from scratch)")
    visualizations.visualize_pooling_scratch(scratch_model, selected_image)
    
    # Backpropagation explanation
    st.subheader("Backpropagation in CNN (from scratch)")
    visualizations.explain_backpropagation_scratch()
    
    # Train model button
    if st.button("Train Model from Scratch (Sample Run)"):
        with st.spinner('Training CNN from scratch...'):
            # Take a small subset for quick demonstration
            X_sample = X_train[:100]
            y_sample = y_train[:100]
            
            # Train the model
            history = scratch_model.train(X_sample, y_sample, epochs=1, batch_size=batch_size)
            
            # Show training results
            st.success('Training completed!')
            st.line_chart(pd.DataFrame({
                'Loss': history['loss']
            }))
            
            # Evaluate on test data
            accuracy = scratch_model.evaluate(X_test[:50], y_test[:50])
            st.metric("Test Accuracy", f"{accuracy:.2%}")

            indices = np.random.choice(len(X_test), 5, replace=False)
            st.subheader("Predictions on Test Images")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    if i < len(indices):
                        index = indices[i]
                        image = X_test[index]
                        label = y_test[index]   
                        pred = scratch_model.predict(image)
                        predicted_class = np.argmax(pred)
                        st.image(np.squeeze(image), caption=f"Predicted Class: {predicted_class}", use_container_width=True)
                        st.write(f"True Class: {label}")
                        st.write(f"Predicted Class: {predicted_class}")
                        

# CNN with Libraries section
elif app_mode == "CNN with Libraries":
    st.header("CNN Implementation with Deep Learning Libraries")
    
    library = st.radio("Select Library", ["TensorFlow", "PyTorch"])
    
    st.markdown(f"""
    This section demonstrates a CNN implemented using {library}. 
    We'll visualize the model architecture and examine how the library handles CNN operations.
    """)
    
    # Load the dataset
    X_train, X_test, y_train, y_test, digits = load_mnist_data()
    
    if library == "TensorFlow":
        # Prepare data for TensorFlow
        tf_model = cnn_tensorflow.TFCNN(
            input_shape=(8, 8, 1),
            num_classes=10,
            conv_filters=conv_filters,
            learning_rate=learning_rate
        )
        
        # Display model architecture
        st.subheader("TensorFlow CNN Architecture")
        tf_model.display_model_summary()
        
        # Visualize layers
        st.subheader("TensorFlow CNN Layer Visualization")
        visualizations.visualize_tf_layers(tf_model, X_test[0:1])
        
        # Train model button
        if st.button("Train TensorFlow Model (Sample Run)"):
            with st.spinner('Training CNN with TensorFlow...'):
                # Take a small subset for quick demonstration
                X_sample = X_train[:700]
                y_sample = y_train[:700]
                
                # Train the model
                history = tf_model.train(X_sample, y_sample, epochs=num_epochs, batch_size=batch_size)
                
                # Show training results
                st.success('Training completed!')
                
                # Plot training history
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(history.history['loss'])
                ax[0].set_title('Model Loss')
                ax[0].set_xlabel('Epoch')
                ax[0].set_ylabel('Loss')
                
                ax[1].plot(history.history['accuracy'])
                ax[1].set_title('Model Accuracy')
                ax[1].set_xlabel('Epoch')
                ax[1].set_ylabel('Accuracy')
                
                st.pyplot(fig)
                
                # Evaluate on test data
                test_loss, test_accuracy = tf_model.evaluate(X_test, y_test)
                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                st.subheader("Predictions on Test Images")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if i < len(X_test):
                            index = i
                            image = X_test[index]
                            label = y_test[index]   
                            pred = tf_model.predict(image.reshape(1, 8, 8, 1))
                            predicted_class = np.argmax(pred)
                            st.image(np.squeeze(image), caption=f"Predicted Class: {predicted_class}", use_container_width=True)
                            st.write(f"True Class: {label}")
                            st.write(f"Predicted Class: {predicted_class}")

    
    else:  # PyTorch
        # Prepare data for PyTorch
        torch_model = cnn_pytorch.TorchCNN(
            input_shape=(1, 8, 8),
            num_classes=10,
            conv_filters=conv_filters,
            learning_rate=learning_rate
        )
        
        # Display model architecture
        st.subheader("PyTorch CNN Architecture")
        st.text(str(torch_model.model))
        
        # Visualize layers
        st.subheader("PyTorch CNN Layer Visualization")
        visualizations.visualize_torch_layers(torch_model, X_test[0])
        
        # Train model button
        if st.button("Train PyTorch Model (Sample Run)"):
            with st.spinner('Training CNN with PyTorch...'):
                # Take a small subset for quick demonstration
                X_sample = X_train[:700]
                y_sample = y_train[:700]
                
                # Train the model
                losses, accuracies = torch_model.train(X_sample, y_sample, epochs=num_epochs, batch_size=batch_size)
                
                # Show training results
                st.success('Training completed!')
                
                # Plot training history
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(losses)
                ax[0].set_title('Model Loss')
                ax[0].set_xlabel('Batch')
                ax[0].set_ylabel('Loss')
                
                ax[1].plot(accuracies)
                ax[1].set_title('Model Accuracy')
                ax[1].set_xlabel('Batch')
                ax[1].set_ylabel('Accuracy')
                
                st.pyplot(fig)
                
                # Evaluate on test data
                test_accuracy = torch_model.evaluate(X_test, y_test)
                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                st.subheader("Predictions on Test Images")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if i < len(X_test):
                            index = i
                            image = X_test[index]
                            label = y_test[index]   
                            pred = torch_model.predict(image)
                            predicted_class = np.argmax(pred)
                            st.image(np.squeeze(image), caption=f"Predicted Class: {predicted_class}", use_container_width=True)
                            st.write(f"True Class: {label}")
                            st.write(f"Predicted Class: {predicted_class}")


# Comparison section
elif app_mode == "Comparison":
    st.header("Comparison: Scratch vs. Library Implementations")
    
    st.markdown("""
    This section provides a side-by-side comparison between the CNN implementation 
    from scratch and using libraries. We'll compare performance, code complexity, 
    and execution speed.
    """)
    
    # Load the dataset
    X_train, X_test, y_train, y_test, digits = load_mnist_data()
    
    # Initialize models
    scratch_model = cnn_scratch.CNN(
        input_shape=(8, 8, 1),
        conv_filters=conv_filters,
        learning_rate=learning_rate
    )
    
    tf_model = cnn_tensorflow.TFCNN(
        input_shape=(8, 8, 1),
        num_classes=10,
        conv_filters=conv_filters,
        learning_rate=learning_rate
    )
    
    torch_model = cnn_pytorch.TorchCNN(
        input_shape=(1, 8, 8),
        num_classes=10,
        conv_filters=conv_filters,
        learning_rate=learning_rate
    )
    
    # Compare forward pass on a single image
    st.subheader("Forward Pass Comparison")
    selected_image_idx = st.slider("Select image for comparison", 0, len(X_test)-1, 0)
    selected_image = X_test[selected_image_idx]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**NumPy (Scratch)**")
        start_time = time.time()
        scratch_out = scratch_model.predict(selected_image)
        scratch_time = time.time() - start_time
        st.bar_chart(pd.DataFrame({'Probability': scratch_out}))
        st.metric("Predicted Class", np.argmax(scratch_out))
        st.metric("Inference Time", f"{scratch_time:.5f} sec")
    
    with col2:
        st.markdown("**TensorFlow**")
        start_time = time.time()
        tf_out = tf_model.predict(selected_image.reshape(1, 8, 8, 1))[0]
        tf_time = time.time() - start_time
        st.bar_chart(pd.DataFrame({'Probability': tf_out}))
        st.metric("Predicted Class", np.argmax(tf_out))
        st.metric("Inference Time", f"{tf_time:.5f} sec")
    
    with col3:
        st.markdown("**PyTorch**")
        start_time = time.time()
        torch_out = torch_model.predict(selected_image)
        torch_time = time.time() - start_time
        st.bar_chart(pd.DataFrame({'Probability': torch_out}))
        st.metric("Predicted Class", np.argmax(torch_out))
        st.metric("Inference Time", f"{torch_time:.5f} sec")
    
    # Code complexity comparison
    st.subheader("Code Complexity Comparison")
    code_comparison = {
        "Implementation": ["Scratch (NumPy)", "TensorFlow", "PyTorch"],
        "Lines of Code (Approx.)": [250, 100, 100],
        "Readability": ["Detailed but complex", "Concise, high-level API", "Pythonic, flexible"],
        "Customizability": ["High", "Medium", "High"],
        "Debugging Ease": ["Straightforward", "Complex (abstractions)", "Good tracebacks"]
    }
    st.table(pd.DataFrame(code_comparison))
    
    # Performance benchmark
    st.subheader("Performance Benchmark")
    if st.button("Run Performance Benchmark"):
        with st.spinner("Running benchmark..."):
            # Prepare small batch for benchmark
            X_batch = X_train[:50]
            y_batch = y_train[:50]
            
            # Benchmark NumPy implementation
            start_time = time.time()
            scratch_model.train(X_batch, y_batch, epochs=1, batch_size=batch_size)
            numpy_time = time.time() - start_time
            
            # Benchmark TensorFlow implementation
            start_time = time.time()
            tf_model.train(X_batch, y_batch, epochs=1, batch_size=batch_size)
            tf_time = time.time() - start_time
            
            # Benchmark PyTorch implementation
            start_time = time.time()
            torch_model.train(X_batch, y_batch, epochs=1, batch_size=batch_size)
            torch_time = time.time() - start_time
            
            # Display benchmark results
            benchmark_data = {
                "Implementation": ["NumPy (Scratch)", "TensorFlow", "PyTorch"],
                "Training Time (sec)": [numpy_time, tf_time, torch_time],
                "Relative Speed": [1.0, numpy_time/tf_time, numpy_time/torch_time]
            }
            st.table(pd.DataFrame(benchmark_data))
            
            # Plot benchmark results
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(benchmark_data["Implementation"], benchmark_data["Training Time (sec)"])
            ax.set_title("Training Time Comparison")
            ax.set_ylabel("Time (seconds)")
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}s', ha='center', va='bottom')
            
            st.pyplot(fig)

# Training Visualization section
elif app_mode == "Training Visualization":
    st.header("CNN Training Visualization")
    
    st.markdown("""
    This section provides a visual representation of the CNN training process,
    including loss and accuracy curves, filter visualizations, and feature maps.
    """)
    
    library = st.radio("Select Implementation", ["NumPy (Scratch)", "TensorFlow", "PyTorch"])
    
    # Load the dataset
    X_train, X_test, y_train, y_test, digits = load_mnist_data()
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        display_lr = st.metric("Learning Rate", learning_rate)
    with col2:
        display_batch = st.metric("Batch Size", batch_size)
    with col3:
        display_epochs = st.metric("Epochs", num_epochs)
    
    # Run training visualization
    if st.button("Start Training Visualization"):
        # Select model based on chosen library
        if library == "NumPy (Scratch)":
            with st.spinner('Training CNN from scratch...'):
                scratch_model = cnn_scratch.CNN(
                    input_shape=(8, 8, 1),
                    conv_filters=conv_filters,
                    learning_rate=learning_rate
                )
                
                # Show architecture
                visualizations.display_scratch_cnn_architecture(scratch_model)
                
                # Setup progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()
                accuracy_chart = st.empty()
                filter_viz = st.empty()
                
                # Training data
                losses = []
                accuracies = []
                
                # Train for specified epochs
                for epoch in range(num_epochs):
                    epoch_losses = []
                    correct_preds = 0
                    total_preds = 0
                    
                    # Process in batches
                    for i in range(0, min(500, len(X_train)), batch_size):
                        X_batch = X_train[i:i+batch_size]
                        y_batch = y_train[i:i+batch_size]
                        
                        # Train on batch
                        batch_loss = scratch_model.train_step(X_batch, y_batch)
                        epoch_losses.append(batch_loss)
                        
                        # Update batch predictions
                        for j in range(len(X_batch)):
                            pred = np.argmax(scratch_model.predict(X_batch[j]))
                            if pred == y_batch[j]:
                                correct_preds += 1
                            total_preds += 1
                        
                        # Calculate current accuracy
                        current_accuracy = correct_preds / total_preds if total_preds > 0 else 0
                        
                        # Update progress
                        progress = (epoch * len(X_train) + i + len(X_batch)) / (num_epochs * min(500, len(X_train)))
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Epoch {epoch+1}/{num_epochs} - Batch {i//batch_size} - Loss: {batch_loss:.4f} - Accuracy: {current_accuracy:.4f}")
                        
                        # Record metrics
                        losses.append(batch_loss)
                        accuracies.append(current_accuracy)
                        
                        # Update charts
                        loss_chart.line_chart(pd.DataFrame({"Loss": losses}))
                        accuracy_chart.line_chart(pd.DataFrame({"Accuracy": accuracies}))
                        
                        # Visualize filters periodically
                        if i % (3 * batch_size) == 0:
                            with filter_viz.container():
                                visualizations.show_scratch_filters(scratch_model)
                
                # Final evaluation
                test_accuracy = scratch_model.evaluate(X_test, y_test)
                st.success(f"Training completed! Final test accuracy: {test_accuracy:.2%}")
                
                # Display feature maps for a sample image
                st.subheader("Feature Maps for Sample Image")
                sample_idx = np.random.randint(0, len(X_test))
                visualizations.show_scratch_feature_maps(scratch_model, X_test[sample_idx])
        
        elif library == "TensorFlow":
            with st.spinner('Training CNN with TensorFlow...'):
                tf_model = cnn_tensorflow.TFCNN(
                    input_shape=(8, 8, 1),
                    num_classes=10,
                    conv_filters=conv_filters,
                    learning_rate=learning_rate
                )
                
                # Show architecture
                tf_model.display_model_summary()
                
                # Prepare callbacks for visualization
                history, model, visualizations_callback = tf_model.train_with_visualization(
                    X_train[:500], y_train[:500], 
                    epochs=num_epochs, 
                    batch_size=batch_size,
                    validation_data=(X_test, y_test)
                )
                
                # Plot final metrics
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(history.history['loss'], label='Training Loss')
                ax[0].plot(history.history['val_loss'], label='Validation Loss')
                ax[0].set_title('Loss Curves')
                ax[0].set_xlabel('Epoch')
                ax[0].set_ylabel('Loss')
                ax[0].legend()
                
                ax[1].plot(history.history['accuracy'], label='Training Accuracy')
                ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax[1].set_title('Accuracy Curves')
                ax[1].set_xlabel('Epoch')
                ax[1].set_ylabel('Accuracy')
                ax[1].legend()
                
                st.pyplot(fig)
                
                # Display final metrics
                test_loss, test_accuracy = tf_model.evaluate(X_test, y_test)
                st.success(f"Training completed! Final test accuracy: {test_accuracy:.2%}")
                
                # Display feature maps
                st.subheader("Feature Maps for Sample Image")
                sample_idx = np.random.randint(0, len(X_test))
                visualizations.show_tf_feature_maps(tf_model, X_test[sample_idx])
                
                # Display learned filters
                st.subheader("Learned Filters")
                visualizations.show_tf_filters(tf_model)
        
        else:  # PyTorch
            with st.spinner('Training CNN with PyTorch...'):
                torch_model = cnn_pytorch.TorchCNN(
                    input_shape=(1, 8, 8),
                    num_classes=10,
                    conv_filters=conv_filters,
                    learning_rate=learning_rate
                )
                
                # Show architecture
                st.text(str(torch_model.model))
                
                # Setup progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()
                accuracy_chart = st.empty()
                filter_viz = st.empty()
                
                # Train with visualization
                losses, accuracies, filter_weights = torch_model.train_with_visualization(
                    X_train[:500], y_train[:500],
                    epochs=num_epochs,
                    batch_size=batch_size,
                    progress_callback=lambda progress, loss, acc: (
                        progress_bar.progress(progress),
                        status_text.text(f"Progress: {progress:.1%} - Loss: {loss:.4f} - Accuracy: {acc:.4f}"),
                        loss_chart.line_chart(pd.DataFrame({"Loss": losses})),
                        accuracy_chart.line_chart(pd.DataFrame({"Accuracy": accuracies})),
                        filter_viz.pyplot(visualizations.plot_torch_filters(filter_weights[-1]))
                        if len(filter_weights) > 0 else None
                    )
                )
                
                # Display final metrics
                test_accuracy = torch_model.evaluate(X_test, y_test)
                st.success(f"Training completed! Final test accuracy: {test_accuracy:.2%}")
                
                # Plot final metrics
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(losses)
                ax[0].set_title('Loss Curve')
                ax[0].set_xlabel('Batch')
                ax[0].set_ylabel('Loss')
                
                ax[1].plot(accuracies)
                ax[1].set_title('Accuracy Curve')
                ax[1].set_xlabel('Batch')
                ax[1].set_ylabel('Accuracy')
                
                st.pyplot(fig)
                
                # Display feature maps
                st.subheader("Feature Maps for Sample Image")
                sample_idx = np.random.randint(0, len(X_test))
                visualizations.show_torch_feature_maps(torch_model, X_test[sample_idx])

# Add a footer
st.markdown("""
    <hr>
    <div style="text-align: center;">
        <small>&copy; 2025 Shivaji Chaulagain. All rights reserved.</small>
    </div>
""", unsafe_allow_html=True)
