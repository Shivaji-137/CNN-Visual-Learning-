import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
import torch

def preprocess_mnist_data(X, y, reshape_for_cnn=True):
    """Preprocess MNIST data for CNN"""
    # Normalize data
    X = X / 255.0
    
    # Reshape for CNN if needed
    if reshape_for_cnn:
        if len(X.shape) == 2:  # If data is flattened
            # Assuming 28x28 images
            img_size = int(np.sqrt(X.shape[1]))
            X = X.reshape(-1, img_size, img_size, 1)
    
    return X, y

def create_confusion_matrix(y_true, y_pred, class_names=None):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    return plt.gcf()

def plot_training_history(history, metrics=['loss', 'accuracy']):
    """Plot training history from tf.keras or custom history dict"""
    fig, axs = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    
    if len(metrics) == 1:
        axs = [axs]
    
    for i, metric in enumerate(metrics):
        if isinstance(history, dict):
            # Handle custom history dict
            if metric in history:
                axs[i].plot(history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history:
                    axs[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
        else:
            # Handle tf.keras history
            if metric in history.history:
                axs[i].plot(history.history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history.history:
                    axs[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        
        axs[i].set_title(f'Model {metric}')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
    
    plt.tight_layout()
    return fig

def compare_execution_time(scratch_fn, tf_fn, torch_fn, *args):
    """Compare execution time of different implementations"""
    import time
    
    # Measure scratch implementation
    start_time = time.time()
    scratch_result = scratch_fn(*args)
    scratch_time = time.time() - start_time
    
    # Measure TensorFlow implementation
    start_time = time.time()
    tf_result = tf_fn(*args)
    tf_time = time.time() - start_time
    
    # Measure PyTorch implementation
    start_time = time.time()
    torch_result = torch_fn(*args)
    torch_time = time.time() - start_time
    
    # Create comparison plot
    implementations = ['Scratch (NumPy)', 'TensorFlow', 'PyTorch']
    times = [scratch_time, tf_time, torch_time]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(implementations, times)
    
    # Add time labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}s', ha='center', va='bottom')
    
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    
    return {
        'scratch': {'result': scratch_result, 'time': scratch_time},
        'tensorflow': {'result': tf_result, 'time': tf_time},
        'pytorch': {'result': torch_result, 'time': torch_time},
        'plot': plt.gcf()
    }

def convert_tf_to_numpy(tensor):
    """Convert TensorFlow tensor to NumPy array"""
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return tensor

def convert_torch_to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor
