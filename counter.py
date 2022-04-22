from ai import Model

model = Model(2, 1)
model.learning_rate(0.001)
inputs = [[1, 1], [1, 2], [2, 2], [4, 7], [10, 10], [32, 57], [1, 57]]
outputs = [[2], [3], [4], [11], [20], [89], [58]]
model.train(np.array(inputs), np.array(outputs), epochs=5000)
output = model.predict(np.array([[2, 57]]))
print(output)
print(round(output[0][0]))
