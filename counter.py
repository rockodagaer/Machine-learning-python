import numpy as np
from ai import Model
# this is an example of how to make an accurate counter using the ai.
# ofcourse it can do much more than this. i made a full calculator with it but you could teach it to do anythind you want

model = Model(1, 1) # the model has 1 input and 1 output so you set it to 1, 1
model.learning_rate(0.001) # set the training data to something low
inputs = [[0], [1], [2], [3], [4], [5], [6], [7]] # inputs, the outputs will be the same item in outputs, warning: it doesnt take lists of lists
outputs = [[1], [2], [3], [4], [5], [6], [7], [8]] # outputs
model.train(np.array(inputs), np.array(outputs), epochs=5000) # train the model with 50000 epcochs will take only a min ( more for better output )
output = model.predict(np.array([[2]])) # predict the output as a float
print(output) # print the output as a float
print(round(output[0][0])) # return it with round() not int() because it relies on that you round the number to make it perfect
