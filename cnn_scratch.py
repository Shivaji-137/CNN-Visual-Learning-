import numpy as np
from tqdm import tqdm

class CNN:
    def __init__(self, input_shape=(8, 8, 1), conv_filters=16, learning_rate=0.01):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.conv_filters = conv_filters
        
        # Initialize CNN layers
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize CNN parameters with random values"""
        # Conv layer parameters (3x3 filters)
        self.conv1_filters = np.random.randn(3, 3, self.input_shape[2], self.conv_filters) * 0.1
        self.conv1_bias = np.zeros(self.conv_filters)
        
        # Calculate output dimensions after convolution and pooling
        self.conv1_output_shape = (
            self.input_shape[0] - 2,  # 3x3 filter reduces dimensions by 2
            self.input_shape[1] - 2,
            self.conv_filters
        )
        
        # Pooling (2x2) dimensions
        self.pool1_output_shape = (
            self.conv1_output_shape[0] // 2,
            self.conv1_output_shape[1] // 2,
            self.conv_filters
        )
        
        # Flatten size
        self.flatten_size = np.prod(self.pool1_output_shape)
        
        # Fully connected layer
        self.fc1_weights = np.random.randn(self.flatten_size, 10) * 0.1
        self.fc1_bias = np.zeros(10)
        
        # Store activations for backpropagation
        self.activations = {}
        
    def convolution(self, input_data, filter_weights, bias):
        """Perform convolution operation"""
        # Get dimensions
        input_height, input_width, input_channels = input_data.shape
        filter_height, filter_width, _, num_filters = filter_weights.shape
        
        # Calculate output dimensions
        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1
        
        # Initialize output
        output = np.zeros((output_height, output_width, num_filters))
        
        # Perform convolution
        for h in range(output_height):
            for w in range(output_width):
                for f in range(num_filters):
                    # Extract the region to apply filter on
                    region = input_data[h:h+filter_height, w:w+filter_width, :]
                    # Perform convolution (element-wise multiplication and sum)
                    output[h, w, f] = np.sum(region * filter_weights[:, :, :, f]) + bias[f]
        
        return output
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def max_pooling(self, input_data, pool_size=2, stride=2):
        """Max pooling operation"""
        # Get dimensions
        input_height, input_width, num_channels = input_data.shape
        
        # Calculate output dimensions
        output_height = input_height // stride
        output_width = input_width // stride
        
        # Initialize output
        output = np.zeros((output_height, output_width, num_channels))
        
        # Perform max pooling
        for h in range(output_height):
            for w in range(output_width):
                h_start, w_start = h * stride, w * stride
                h_end, w_end = h_start + pool_size, w_start + pool_size
                
                # Extract the region
                region = input_data[h_start:h_end, w_start:w_end, :]
                
                # Take maximum value
                output[h, w, :] = np.max(region, axis=(0, 1))
        
        return output
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """Forward propagation through the network"""
        # Store input
        self.activations['input'] = X
        
        # Convolution layer
        self.activations['conv1'] = self.convolution(X, self.conv1_filters, self.conv1_bias)
        
        # ReLU activation
        self.activations['relu1'] = self.relu(self.activations['conv1'])
        
        # Max pooling
        self.activations['pool1'] = self.max_pooling(self.activations['relu1'])
        
        # Flatten
        flattened = self.activations['pool1'].reshape(-1, self.flatten_size)
        self.activations['flatten'] = flattened
        
        # Fully connected layer
        fc1_output = np.dot(flattened, self.fc1_weights) + self.fc1_bias
        self.activations['fc1'] = fc1_output
        
        # Softmax
        output = self.softmax(fc1_output)
        self.activations['output'] = output
        
        return output
    
    def backward_pass(self, y):
        """Backward propagation to update weights"""
        # Number of samples
        m = 1
        
        # Calculate the gradient of the output layer
        # For softmax with cross-entropy, gradient is (output - y_one_hot)
        y_one_hot = np.zeros(10)
        y_one_hot[y] = 1
        dout = self.activations['output'] - y_one_hot.reshape(1, -1)
        
        # Gradient for fully connected layer
        dW_fc1 = np.dot(self.activations['flatten'].T, dout)
        db_fc1 = np.sum(dout, axis=0)
        
        # Gradient for flatten layer
        dflat = np.dot(dout, self.fc1_weights.T)
        
        # Reshape gradient to match pool1 output shape
        dpool = dflat.reshape(self.activations['pool1'].shape)
        
        # Gradient for max pooling layer (upsampling)
        drelu = np.zeros(self.activations['relu1'].shape)
        pool_size = 2
        
        # Loop through each pooling region to propagate the gradient
        for h in range(dpool.shape[0]):
            for w in range(dpool.shape[1]):
                for c in range(dpool.shape[2]):
                    h_start, w_start = h * pool_size, w * pool_size
                    h_end, w_end = h_start + pool_size, w_start + pool_size
                    
                    # Extract the region
                    region = self.activations['relu1'][h_start:h_end, w_start:w_end, c]
                    
                    # Find the max value index
                    mask = (region == np.max(region))
                    
                    # Update gradient at max value position
                    drelu[h_start:h_end, w_start:w_end, c] += mask * dpool[h, w, c]
        
        # Gradient for ReLU activation
        drelu[self.activations['relu1'] <= 0] = 0
        
        # Gradient for convolution layer
        # Initialize gradient for filters and bias
        dW_conv1 = np.zeros_like(self.conv1_filters)
        db_conv1 = np.zeros_like(self.conv1_bias)
        
        # Calculate gradients for convolution operation
        for h in range(drelu.shape[0]):
            for w in range(drelu.shape[1]):
                for f in range(drelu.shape[2]):
                    # Extract the region
                    region = self.activations['input'][h:h+3, w:w+3, :]
                    
                    # Update gradient for filter
                    dW_conv1[:, :, :, f] += region * drelu[h, w, f]
                    
                    # Update gradient for bias
                    db_conv1[f] += drelu[h, w, f]
        
        # Update weights and biases
        self.conv1_filters -= self.learning_rate * dW_conv1
        self.conv1_bias -= self.learning_rate * db_conv1
        self.fc1_weights -= self.learning_rate * dW_fc1
        self.fc1_bias -= self.learning_rate * db_fc1
    
    def train_step(self, X_batch, y_batch):
        """Train network on a single batch"""
        total_loss = 0
        
        for i in range(len(X_batch)):
            # Forward pass
            output = self.forward_pass(X_batch[i])
            
            # Calculate cross-entropy loss
            y_one_hot = np.zeros(10)
            y_one_hot[y_batch[i]] = 1
            loss = -np.sum(y_one_hot * np.log(output + 1e-15))
            total_loss += loss
            
            # Backward pass
            self.backward_pass(y_batch[i])
        
        return total_loss / len(X_batch)
    
    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """Train network on the entire dataset"""
        num_samples = len(X_train)
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Process in batches
            total_loss = 0
            for i in range(0, num_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Train on batch
                batch_loss = 0
                for j in range(len(X_batch)):
                    # Forward pass
                    output = self.forward_pass(X_batch[j])
                    
                    # Calculate cross-entropy loss
                    y_one_hot = np.zeros(10)
                    y_one_hot[y_batch[j]] = 1
                    loss = -np.sum(y_one_hot * np.log(output + 1e-15))
                    batch_loss += loss
                    
                    # Backward pass
                    self.backward_pass(y_batch[j])
                
                avg_batch_loss = batch_loss / len(X_batch)
                total_loss += avg_batch_loss
            
            # Record average loss for epoch
            avg_loss = total_loss / (num_samples // batch_size)
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make prediction for a single sample"""
        output = self.forward_pass(X)
        return output[0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        correct = 0
        for i in range(len(X_test)):
            # Forward pass
            output = self.predict(X_test[i])
            
            # Get predicted class
            predicted_class = np.argmax(output)
            
            # Check if prediction is correct
            if predicted_class == y_test[i]:
                correct += 1
        
        # Calculate accuracy
        accuracy = correct / len(X_test)
        return accuracy