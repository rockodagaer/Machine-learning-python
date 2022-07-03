# Machine-learning-python
With tenosrflow you need to do thousands of things while with this you just imediatly train it with some in and outputs and its done.

Items used: -------------------------------------

- github copilot
- gpt-3 Codex
- visual studio code
- python version 3.10.5
- numpy

Change Log: -------------------------------------

- new training methods: added item training, a faster version of smart training
- getting data: built in method for getting data from json files

How to use: ( NEW ITEMS ) -----------------------

data_get is made so you can get data from a json file
you use it like data_get("data.json") and the data inside if the file looks somewhat like:
[ [[0], [1]], [[1], [2]] ] of this list are sections with different inputs and outputs
the first item in each of these sections are the input and the second item is the output

train_items is a training method that goes through every item and trains it with the set amount of epochs
this is usefull if you want to get VERY specivic outputs, this means it has 0 creativity
and its used like: train_items(self, data="data.json", epochs=5000) this one uses data_get formats so remember that

train_items_smart is the same as train_items but it uses train_smart instead of train which makes it take longer but even more specivec
here you can see an example of how its used: train_items_smart(self, "data.json", epochs=500, amount=0.0001, progress=False, show=False, retrain=0)
this also uses data_get, amount is the needed error rate to pass and this should be WAY under 0, progress shows its progress and show shows the epochs per sec.

