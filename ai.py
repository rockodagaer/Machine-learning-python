import numpy as np
import json
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
        self.layers = {} # create a list of layers
    
    def predict(self, inputs): # predict a number
        return np.dot(inputs, self.weights) + self.bias # return the predicted output (numbers only)

    def train(self, inputs, expected_outputs, epochs=1, progress=False): # train the model
        # inputs and expected_outputs are lists like [[0], [1], [2]]
        # epochs is the amount of times it trains the model
        inputs = np.array(inputs) # convert to numpy array
        expected_outputs = np.array(expected_outputs) # convert to numpy array
        for epoch in range(epochs):
            predicted_outputs = self.predict(inputs) # predict the inputs now
            error = predicted_outputs - expected_outputs # get the error amount
            gradient = np.dot(inputs.T, error)
            gradient /= inputs.shape[0]
            self.weights -= self.learning_rate * gradient # set the weights to the updated amount
            self.bias -= self.learning_rate * np.mean(error, axis=0) # set bias to the updated amount
            if progress: print(self.predict(inputs))

    def train_smart(self, inputs, expected_outputs, epochs=1, amount=0.0001, progress=False, retrain=0): # train until model is perfect
        # train and test until the error is low enough
        start = time.time() # start timer
        done = 0
        while True: # loop
            done += epochs
            self.train(inputs, expected_outputs, epochs) # train the model
            predicted_outputs = self.predict(inputs) # predict
            if progress:
                # print how many epochs per second
                print("Epochs per sec: " + str(done / (time.time() - start)), end="\r")
                if time.time() - start > 1:
                    done = 0
                    start = time.time()
            error = predicted_outputs - expected_outputs # check the error
            if np.mean(np.abs(error)) < amount: # check if error is low enough
                break
        if retrain > 0 and self.test(inputs, expected_outputs) > amount / 2: # if error is too high retrain
            for i in range(retrain):
                self.train_smart(inputs, expected_outputs, epochs, amount, progress)
        if progress: print()
    
    def train_fast(self, inputs, expected_outputs, epochs=1, timer=0.01, progress=False): # train for specified amount of time
        # train until it reaches the specified amount of epochs or it reaches the max time
        start = time.time() # start timer
        done = 0
        for epoch in range(epochs): # loop
            done += 1
            self.train(inputs, expected_outputs, epochs=1) # train 1 epoch
            if progress:
                # print how many epochs per second
                print("Epochs per sec: " + str(done / (time.time() - start)), end="\r")
                if time.time() - start > timer:
                    done = 0
                    start = time.time()
            if time.time() - start > timer: # if takes too long stop the loop
                break
        if progress: print()
    
    def train_items(self, data, epochs=1, progress=False): # train everything in the data individually
        inputs = self.data_get(data)[0]
        outputs = self.data_get(data)[1]
        for i in range(len(inputs)):
            self.train(inputs[i], outputs[i], epochs)
            if progress: print("Input Item: " + str(inputs[i]), end=" "), print("Output Item: " + str(outputs[i]), end="\r")
    
    def train_items_smart(self, data, epochs=1, amount=0.0001, progress=False, show=False, show_length=1, retrain=0): # train everything in the data individually but using train_smart
        inputs = self.data_get(data)[0]
        outputs = self.data_get(data)[1]
        for i in range(len(inputs)):
            self.train_smart(inputs[i], outputs[i], epochs, amount, progress)
            if progress: print(self.test(inputs[i], outputs[i]))
            if show:
                if show_length > len(inputs):
                    show_length = len(inputs)
                for i in range(show_length):
                    print("Predicted: " + str(inputs[i]) + " - " + str(self.predict(inputs[i])))
        if retrain > 0 and self.test(inputs, outputs) > amount / 2:
            for i in range(retrain):
                if self.test(inputs, outputs) > amount / (1000 * 1000): self.train_items_smart(data, epochs, amount=amount, progress=progress, show=show, show_length=show_length)

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
    
    def data_get(self, data):
        f = open(data, "r")
        data = json.load(f)
        f.close()
        inputs = []
        outputs = []
        for i in range(len(data)):
            inputs.append(data[i][0])
            outputs.append(data[i][1])
        return [inputs, outputs]
