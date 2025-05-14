import numpy as np

class CNN:
    def __init__(self, input_shape=(8, 8, 1), conv_filters=4, learning_rate=0.01, num_classes=10):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.init_weights()

    def init_weights(self):
        # Convolutional layer (3x3 filter)
        self.W_conv = np.random.randn(3, 3, self.input_shape[2], self.conv_filters) * 0.1
        self.b_conv = np.zeros(self.conv_filters)

        # Compute output shape after conv (no padding)
        conv_out_h = self.input_shape[0] - 2
        conv_out_w = self.input_shape[1] - 2

        # Max pooling 2x2
        pool_out_h = conv_out_h // 2
        pool_out_w = conv_out_w // 2

        self.flatten_size = pool_out_h * pool_out_w * self.conv_filters

        # Fully connected layer
        self.W_fc = np.random.randn(self.flatten_size, self.num_classes) * np.sqrt(2. / self.flatten_size)
        self.b_fc = np.zeros(self.num_classes)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        self.x_input = x  # (8, 8, 1)

        # Convolution
        self.conv_out = np.zeros((6, 6, self.conv_filters))
        for f in range(self.conv_filters):
            for i in range(6):
                for j in range(6):
                    region = x[i:i+3, j:j+3, :]
                    self.conv_out[i, j, f] = np.sum(region * self.W_conv[:, :, :, f]) + self.b_conv[f]
        self.relu_out = self.relu(self.conv_out)

        # Max pooling 2x2
        self.pool_out = np.zeros((3, 3, self.conv_filters))
        self.pool_mask = np.zeros_like(self.relu_out)
        for f in range(self.conv_filters):
            for i in range(3):
                for j in range(3):
                    h_start, w_start = i*2, j*2
                    region = self.relu_out[h_start:h_start+2, w_start:w_start+2, f]
                    max_val = np.max(region)
                    self.pool_out[i, j, f] = max_val
                    mask = (region == max_val)
                    self.pool_mask[h_start:h_start+2, w_start:w_start+2, f] = mask

        self.flatten = self.pool_out.flatten().reshape(1, -1)
        self.logits = self.flatten @ self.W_fc + self.b_fc
        self.probs = self.softmax(self.logits)

        return self.probs

    def backward(self, y_true):
        # One-hot encoding
        y = np.zeros((1, self.num_classes))
        y[0, y_true] = 1

        # Loss derivative
        dL_dlogits = self.probs - y  # (1, 10)

        # Grad for FC
        dW_fc = self.flatten.T @ dL_dlogits
        db_fc = dL_dlogits[0]

        # Backprop to pooling layer
        d_flatten = dL_dlogits @ self.W_fc.T
        d_pool = d_flatten.reshape(self.pool_out.shape)

        # Max pooling backprop
        d_relu = np.zeros_like(self.relu_out)
        for f in range(self.conv_filters):
            for i in range(3):
                for j in range(3):
                    h_start, w_start = i*2, j*2
                    d_relu[h_start:h_start+2, w_start:w_start+2, f] += \
                        self.pool_mask[h_start:h_start+2, w_start:w_start+2, f] * d_pool[i, j, f]

        # ReLU backprop
        d_conv = d_relu * self.relu_derivative(self.conv_out)

        # Grad for conv
        dW_conv = np.zeros_like(self.W_conv)
        db_conv = np.zeros_like(self.b_conv)
        for f in range(self.conv_filters):
            for i in range(6):
                for j in range(6):
                    region = self.x_input[i:i+3, j:j+3, :]
                    dW_conv[:, :, :, f] += d_conv[i, j, f] * region
                    db_conv[f] += d_conv[i, j, f]

        # Update weights
        self.W_fc -= self.learning_rate * dW_fc
        self.b_fc -= self.learning_rate * db_fc
        self.W_conv -= self.learning_rate * dW_conv
        self.b_conv -= self.learning_rate * db_conv

    def train(self, X, y, epochs=10, batch_size=32):
        n = len(X)
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X, y = X[indices], y[indices]
            loss = 0
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                for xi, yi in zip(X_batch, y_batch):
                    output = self.forward(xi)
                    loss += -np.log(output[0, yi] + 1e-9)
                    self.backward(yi)
            avg_loss = loss / n
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, x):
        return np.argmax(self.forward(x))

    def evaluate(self, X, y):
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == y[i]:
                correct += 1
        return correct / len(X)
