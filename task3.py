import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
n_inputs = 784
n_hidden = 256
n_outputs = 10

# Initialize weights and biases
weights1 = np.random.rand(n_inputs, n_hidden) - 0.5
weights2 = np.random.rand(n_hidden, n_outputs) - 0.5
bias1 = np.zeros((1, n_hidden))
bias2 = np.zeros((1, n_outputs))

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# Define the forward pass
def forward_pass(X, weights1, weights2, bias1, bias2):
    hidden_layer = relu(np.dot(X, weights1) + bias1)
    output_layer = softmax(np.dot(hidden_layer, weights2) + bias2)
    return hidden_layer, output_layer

# Define the backward pass
def backward_pass(X, y, hidden_layer, output_layer, weights1, weights2, bias1, bias2):
    d_output_layer = output_layer - y
    d_hidden_layer = d_output_layer * (hidden_layer * (1 - hidden_layer))
    d_weights2 = np.dot(hidden_layer.T, d_output_layer)
    d_bias2 = np.sum(d_output_layer, axis=0, keepdims=True)
    d_weights1 = np.dot(X.T, d_hidden_layer)
    d_bias1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
    return d_weights1, d_bias1, d_weights2, d_bias2

# Define the update rules
def update_weights(weights, bias, d_weights, d_bias, learning_rate):
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
    return weights, bias

# Train the neural network
learning_rate = 0.01
n_epochs = 10
for epoch in range(n_epochs):
    hidden_layer, output_layer = forward_pass(X_train, weights1, weights2, bias1, bias2)
    d_weights1, d_bias1, d_weights2, d_bias2 = backward_pass(X_train, y_train, hidden_layer, output_layer, weights1, weights2, bias1, bias2)
    weights1, bias1 = update_weights(weights1, bias1, d_weights1, d_bias1, learning_rate)
    weights2, bias2 = update_weights(weights2, bias2, d_weights2, d_bias2, learning_rate)

    # Evaluate the model on the test set
    hidden_layer, output_layer = forward_pass(X_test, weights1, weights2, bias1, bias2)
    accuracy = np.mean(np.argmax(output_layer, axis=1) == y_test)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.3f}")

# Use the trained model to make predictions on the test set
hidden_layer, output_layer = forward_pass(X_test, weights1, weights2, bias1, bias2)
predictions = np.argmax(output_layer, axis=1)

# Evaluate the model on the test set
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.3f}")