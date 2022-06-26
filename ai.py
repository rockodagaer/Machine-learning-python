import numpy as np
import time

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
        # inputs and expected_outputs are lists like [[0], [1], [2]]
        # epochs is the amount of times it trains the model
        for epoch in range(epochs):
            predicted_outputs = self.predict(inputs) # predict the inputs now
            error = predicted_outputs - expected_outputs # get the error amount
            gradient = np.dot(inputs.T, error)
            gradient /= inputs.shape[0]
            self.weights -= self.learning_rate * gradient # set the weights to the updated amount
            self.bias -= self.learning_rate * np.mean(error, axis=0) # set bias to the updated amount

    def train_smart(self, inputs, expected_outputs, epochs=1, need=0.0001): # train until model is perfect
        # train and test until the error is low enough
        while True: # loop
            self.train(inputs, expected_outputs, epochs) # train the model
            predicted_outputs = self.predict(inputs) # predict
            error = predicted_outputs - expected_outputs # check the error
            if np.mean(np.abs(error)) < need: # check if error is low enough
                break
    
    def train_fast(self, inputs, expected_outputs, epochs=1, time=0.01): # train for specified amount of time
        # train until it reaches the specified amount of epochs or it reaches the max time
        start = time.time() # start timer
        for epoch in range(epochs): # loop
            self.train(inputs, expected_outputs, epochs=1) # train 1 epoch
            if time.time() - start > time: # if takes too long stop the loop
                break

    def test(self, inputs=[], expected_outputs=[]): # test how good it is at predicting the right outputs
        predicted_outputs = self.predict(inputs)
        error = predicted_outputs - expected_outputs
        return np.mean(np.abs(error)) # return the amount of errors ( warning it doesnt return the exact amount of errors )
    
    def learn_rate(self, learning_rate=0.001): # set learning rate
        self.learning_rate = learning_rate
    
    def save(self, filename): # save your model
        np.save(filename, self.weights)
        np.save(filename + "b", self.bias)
    
    def load(self, filename): # load your model
        self.weights = np.load(filename)
        self.bias = np.load(filename + "b")
