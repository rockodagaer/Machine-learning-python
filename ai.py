import numpy as np

class Model:
    # make a model that takes inputs and outputs and generates a new output based on the inputs and outputs
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)
        self.bias = np.random.rand(outputs)
        self.learning_rate = 0.1 # make this REALLY low to make the ai more accurate
    
    def predict(self, inputs): # predicts the based on your own input questions
        return np.dot(inputs, self.weights) + self.bias

    def train(self, inputs, expected_outputs):
        predicted_outputs = self.predict(inputs)
        error = predicted_outputs - expected_outputs
        self.weights -= self.learning_rate * np.dot(inputs.T, error)
        self.bias -= self.learning_rate * np.sum(error, axis=0)
        
    # mostly unused:
    def test(self, inputs, expected_outputs):
        predicted_outputs = self.predict(inputs)
        error = predicted_outputs - expected_outputs
        return np.mean(np.abs(error))
