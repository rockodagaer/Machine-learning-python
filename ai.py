import numpy as np

class Model:
    # make a model that takes inputs and outputs and generates a new output based on the inputs and outputs
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)
        self.bias = np.random.rand(outputs)
        self.learning_rate = 0.001
    
    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def train(self, inputs, expected_outputs, epochs=1):
        for epoch in range(epochs):
            predicted_outputs = self.predict(inputs)
            error = predicted_outputs - expected_outputs
            gradient = np.dot(inputs.T, error)
            gradient /= inputs.shape[0]
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(error, axis=0)

    def test(self, inputs, expected_outputs):
        predicted_outputs = self.predict(inputs)
        error = predicted_outputs - expected_outputs
        return np.mean(np.abs(error))
    
    def learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
