import numpy as np
from ai import Model
# this is an example of how to make an accurate counter using the ai.
# ofcourse it can do much more than this. i made a full calculator with it but you could teach it to do anythind you want

model = Model(1, 1) # the model has 1 input and 1 output so you set it to 1, 1
inputs = model.data_get("data.json")[0]
outputs = model.data_get("data.json")[1]
model.train_items_smart("data.json", epochs=500, amount=0.000000001, retrain=50, progress=True, show=True)
output = model.predict([[69]])
print(output)
print(round(output[0][0]))