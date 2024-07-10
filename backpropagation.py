import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))
    else:
        return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    hidden_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
    output_weights = np.random.uniform(-1, 1, (hidden_size, output_size))
    return hidden_weights, output_weights


def train(X, y, hidden_weights, output_weights, num_iterations=10000, alpha=0.1):
    for i in range(num_iterations):
        input_layer_outputs = X
        hidden_layer_inputs = np.dot(input_layer_outputs, hidden_weights)
        hidden_layer_outputs = sigmoid(hidden_layer_inputs)
        output_layer_inputs = np.dot(hidden_layer_outputs, output_weights)
        output_layer_outputs = sigmoid(output_layer_inputs)

        output_error = output_layer_outputs - y

        hidden_error = sigmoid(hidden_layer_outputs, derivative=True) * np.dot(output_error, output_weights.T)

        hidden_pd = np.dot(input_layer_outputs.T, hidden_error)
        output_pd = np.dot(hidden_layer_outputs.T, output_error)

        hidden_weights -= alpha * hidden_pd
        output_weights -= alpha * output_pd

    return hidden_weights, output_weights


def predict(X, hidden_weights, output_weights):
    hidden_layer_inputs = np.dot(X, hidden_weights)
    hidden_layer_outputs = sigmoid(hidden_layer_inputs)
    output_layer_inputs = np.dot(hidden_layer_outputs, output_weights)
    output_layer_outputs = sigmoid(output_layer_inputs)
    return output_layer_outputs, (output_layer_outputs >= 0.5).astype(int)
