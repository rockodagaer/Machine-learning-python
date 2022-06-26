import numpy as np

# this is old and does not resemble anything special
# it only takes numbers. if you want it to generate text or something you are gonna have to convert text to numbers and numbers to text
# some of this was also made of the closed beta version of github copilot which released a full version now
class Model: # the main ai
    def __init__(self, inputs, outputs):
        self.inputs = inputs # input size
        self.outputs = outputs # output size
        self.weights = np.random.rand(inputs, outputs) # make weights
        self.bias = np.random.rand(outputs) # make biases
        self.learning_rate = 0.001 # set learning rate (the lower the more precise it learns)
    
    def predict(self, inputs): # predict a number
        return np.dot(inputs, self.weights) + self.bias # return the predicted output (numbers only)

    def train(self, inputs, expected_outputs, epochs=1): # train the model
        for epoch in range(epochs):
            predicted_outputs = self.predict(inputs)
            error = predicted_outputs - expected_outputs
            gradient = np.dot(inputs.T, error)
            gradient /= inputs.shape[0]
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(error, axis=0)

    def test(self, inputs, expected_outputs): # test how good it is at predicting the right outputs
        predicted_outputs = self.predict(inputs)
        error = predicted_outputs - expected_outputs
        return np.mean(np.abs(error)) # return the amount of errors ( warning it doesnt return the exact amount of errors )
    
    def learn_rate(self, learning_rate): # set learning rate
        self.learning_rate = learning_rate
