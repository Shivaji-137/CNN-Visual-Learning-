# CNN Visual Learning

This project is an interactive educational application for understanding Convolutional Neural Networks (CNNs). It provides visualizations, comparisons, and hands-on demonstrations of CNNs implemented from scratch and using popular deep learning libraries like TensorFlow and PyTorch.

## Features

- **Interactive Tutorials**: Step-by-step guides to understand CNN concepts.
- **CNN Implementations**:
  - From scratch using NumPy.
  - Using TensorFlow and PyTorch.
- **Visualization Tools**:
  - CNN architecture visualization.
  - Convolution and pooling operations.
  - Training process visualization (loss, accuracy, filters, and feature maps).
- **Comparison**:
  - Performance and code complexity comparison between implementations.
- **Custom Filters**: Build and apply your own convolutional filters.
- **Dataset Support**:
  - Preloaded datasets like MNIST.
  - Option to upload custom images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shivaji-137/CNN-Visual-Learning-.git
   cd CNN-Visual-Learning-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run full_apps.py
   ```

## Project Structure

```
.
├── advanced_visualizations.py   # Advanced visualization utilities
├── architecture_builder.py      # Tools for building CNN architectures
├── classification_playground.py # Interactive classification demos
├── cnn_pytorch.py               # PyTorch CNN implementation
├── cnn_scratch.py               # NumPy-based CNN implementation
├── cnn_tensorflow.py            # TensorFlow CNN implementation
├── complete_cnn_copy.py         # Main application logic
├── filters.py                   # Predefined and custom filters
├── full_apps.py                 # Streamlit application entry point
├── guided_tutorials.py          # Step-by-step tutorials
├── training_visualization.py    # Training visualization utilities
├── utils.py                     # Helper functions
├── visualizations.py            # Visualization utilities
├── LICENSE                      # License file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
```

## Usage

1. Launch the application using Streamlit.
2. Use the sidebar to navigate between sections:
   - **Introduction**: Overview of CNNs.your-username
   - **CNN from Scratch**: Explore a NumPy-based implementation.
   - **Process Sample Images**: Apply filters to sample images.
   - **Upload Your Own Image**: Test CNNs on custom images.
   - **Custom Filter Builder**: Design and apply custom filters.
   - **CNN with Libraries**: Explore TensorFlow and PyTorch implementations.
   - **Comparison**: Compare implementations.
   - **Training Visualization**: Visualize the training process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed by Shivaji Chaulagain.
- Inspired by the need for interactive and visual learning tools for deep learning concepts.

- If you use this code, please give credit to the author.
