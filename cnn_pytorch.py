import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class PyTorchCNN(nn.Module):
    def __init__(self, input_shape, num_filters, num_classes):
        super(PyTorchCNN, self).__init__()
        # Unpack input shape
        channels, height, width = input_shape
        
        # CNN architecture
        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after convolution and pooling
        conv_height = height - 2  # 3x3 kernel reduces height by 2
        conv_width = width - 2    # 3x3 kernel reduces width by 2
        pool_height = conv_height // 2
        pool_width = conv_width // 2
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters * pool_height * pool_width, num_classes)
    
    def forward(self, x):
        # Convolutional layer with ReLU activation
        x = F.relu(self.conv1(x))
        
        # Max pooling
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer with softmax activation
        x = F.softmax(self.fc(x), dim=1)
        
        return x

class TorchCNN:
    def __init__(self, input_shape=(1, 8, 8), num_classes=10, conv_filters=16, learning_rate=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = PyTorchCNN(input_shape, conv_filters, num_classes).to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """Train the model"""
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X_train.reshape(-1, 1, 8, 8), dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Lists to store metrics
        losses = []
        accuracies = []
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                # Move tensors to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store metrics
                losses.append(loss.item())
                accuracies.append(correct / total)
            
            # Print epoch statistics
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        return losses, accuracies
    
    def train_with_visualization(self, X_train, y_train, epochs=5, batch_size=32, progress_callback=None):
        """Train with visualization callback"""
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X_train.reshape(-1, 1, 8, 8), dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Lists to store metrics
        losses = []
        accuracies = []
        filter_weights = []
        
        # Total number of batches
        total_batches = len(dataloader) * epochs
        batch_count = 0
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                # Move tensors to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store metrics
                current_loss = loss.item()
                current_acc = correct / total
                losses.append(current_loss)
                accuracies.append(current_acc)
                
                # Increment batch counter
                batch_count += 1
                progress = batch_count / total_batches
                
                # Store filter weights periodically
                if batch_count % 5 == 0 or batch_count == total_batches:
                    # Get filter weights from first conv layer
                    filter_weights.append(self.model.conv1.weight.detach().cpu().numpy())
                
                # Call progress callback
                if progress_callback:
                    progress_callback(progress, current_loss, current_acc)
            
            # Print epoch statistics
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        return losses, accuracies, filter_weights
    
    def predict(self, X):
        """Make prediction"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Convert input to PyTorch tensor
        X_tensor = torch.tensor(X.reshape(1, 1, 8, 8), dtype=torch.float32).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(X_tensor)
        
        # Convert to numpy array
        return output.cpu().numpy()[0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X_test.reshape(-1, 1, 8, 8), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Track statistics
        correct = 0
        total = 0
        
        # Evaluation loop
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Move tensors to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        accuracy = correct / total
        return accuracy
    
    def get_feature_maps(self, X):
        """Extract feature maps from model for visualization"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Convert input to PyTorch tensor
        X_tensor = torch.tensor(X.reshape(1, 1, 8, 8), dtype=torch.float32).to(self.device)
        
        # Get feature maps
        with torch.no_grad():
            # Forward pass through conv layer
            feature_maps = F.relu(self.model.conv1(X_tensor))
        
        # Convert to numpy array
        return feature_maps[0].cpu().numpy()
    
    def get_filters(self):
        """Extract filters from first convolutional layer"""
        # Get weights from conv1 layer
        filters = self.model.conv1.weight.detach().cpu().numpy()
        return filters
