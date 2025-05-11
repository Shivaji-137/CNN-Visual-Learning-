import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import streamlit as st

class TFVisualizationCallback(Callback):
    def __init__(self, validation_data=None):
        super(TFVisualizationCallback, self).__init__()
        self.validation_data = validation_data
        self.epoch_loss = []
        self.epoch_accuracy = []
        self.batch_loss = []
        self.batch_accuracy = []
        self.filters = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_loss.append(logs.get('loss'))
        self.epoch_accuracy.append(logs.get('accuracy'))
        
        # Store filter weights
        conv_layer = None
        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                conv_layer = layer
                break
        
        if conv_layer is not None:
            weights = conv_layer.get_weights()[0]
            self.filters.append(weights)
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_loss.append(logs.get('loss'))
        self.batch_accuracy.append(logs.get('accuracy'))

class TFCNN:
    def __init__(self, input_shape=(8, 8, 1), num_classes=10, conv_filters=16, learning_rate=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = self.build_model()
    
    def build_model(self):
        """Build CNN model using TensorFlow Keras"""
        model = Sequential([
            # Convolutional layer
            Conv2D(self.conv_filters, kernel_size=(3, 3), activation='relu', 
                   input_shape=self.input_shape, padding='valid'),
            
            # Max pooling layer
            MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten layer
            Flatten(),
            
            # Fully connected layer
            Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def display_model_summary(self):
        """Display model summary"""
        # Convert model summary to string
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        st.text(model_summary)
    
    def train(self, X_train, y_train, epochs=5, batch_size=32, validation_data=None):
        """Train the model"""
        # Convert labels to one-hot encoding
        y_train_categorical = to_categorical(y_train, self.num_classes)
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_categorical = to_categorical(y_val, self.num_classes)
            validation_data = (X_val, y_val_categorical)
        
        # Train model
        history = self.model.fit(
            X_train, y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        return history
    
    def train_with_visualization(self, X_train, y_train, epochs=5, batch_size=32, validation_data=None):
        """Train the model with visualization callback"""
        # Convert labels to one-hot encoding
        y_train_categorical = to_categorical(y_train, self.num_classes)
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_categorical = to_categorical(y_val, self.num_classes)
            validation_data = (X_val, y_val_categorical)
        
        # Create visualization callback
        viz_callback = TFVisualizationCallback(validation_data)
        
        # Train model
        history = self.model.fit(
            X_train, y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[viz_callback],
            verbose=1
        )
        
        return history, self.model, viz_callback
    
    def predict(self, X):
        """Make prediction"""
        # Ensure input is a batch
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=0)
        
        # Make prediction
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        # Convert labels to one-hot encoding
        y_test_categorical = to_categorical(y_test, self.num_classes)
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        return loss, accuracy
    
    def get_feature_maps(self, X):
        """Extract feature maps from model for visualization"""
        # Create a model that outputs feature maps from convolutional layer
        conv_layer = None
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, Conv2D):
                conv_layer = i
                break
        
        if conv_layer is not None:
            feature_map_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=self.model.layers[conv_layer].output
            )
            
            # Ensure input is a batch
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=0)
            
            # Get feature maps
            feature_maps = feature_map_model.predict(X)
            return feature_maps[0]
        
        return None
    
    def get_filters(self):
        """Extract convolutional filters from model"""
        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                # Get weights [filters, bias]
                weights = layer.get_weights()
                return weights[0]
        
        return None
