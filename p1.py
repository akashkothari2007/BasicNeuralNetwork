import numpy as np
np.random.seed(0)


#INPUTS
X = np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])
inputLength = 2
batchSize = 4

#Defining hidden layers
numOfLayers = 2
hiddenLayerSize = [3, 3]





class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.5*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output=np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients of weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


##Activation functions
def sigmoid(output):
    return 1 / (1 + np.exp(-output))   
def sigmoid_derivative(output):
    return output * (1 - output) 


#Initializing layers
layers = []
firstLayer = Layer_Dense(inputLength, hiddenLayerSize[0])
layers.append(firstLayer)
for x in range (numOfLayers - 1):
    newLayer = Layer_Dense(hiddenLayerSize[x], hiddenLayerSize[x + 1])
    layers.append(newLayer)

#End layer
numOfDesiredOutputs = 1


desiredOutputs = np.array([[0],
                           [1],
                           [1],
                           [0]])

finalLayer = Layer_Dense(hiddenLayerSize[numOfLayers - 1], numOfDesiredOutputs)
layers.append(finalLayer)


# Interactive testing loop before learning
while True:
    try:
        # Get inputs
        a = float(input("\nEnter first input (0 or 1, or -1 to quit): "))
        if a == -1:
            break
        b = float(input("Enter second input (0 or 1): "))
        x_test = np.array([[a, b]])

        inputs = x_test
        for layer in layers:
            layer.forward(inputs)
            layer.output = sigmoid(layer.output)
            inputs = layer.output

        result = layers[-1].output[0][0]
        print(f"Prediction: {result:.4f} (interpreted as {'1' if result > 0.5 else '0'})")

    except KeyboardInterrupt:
        break


learning_rate = 1.0
epochs = 5000

for epoch in range(epochs):
    # forward pass
    inputs = X
    for layer in layers:
        layer.forward(inputs)
        layer.output = sigmoid(layer.output)
        inputs = layer.output  # move forward

    predictions = layers[-1].output
    loss = np.mean((desiredOutputs - predictions) ** 2)

    # goes backward 
    dvalues = -2 * (desiredOutputs - predictions) / batchSize

    for layer_i in reversed(range(len(layers))):
        layer = layers[layer_i]
        # Derivative of activation
        dactivation = dvalues * sigmoid_derivative(layer.output)
        layer.backward(dactivation)
        dvalues = layer.dinputs  # pass gradients back

    # update weights based on derivatives
    for layer in layers:
        layer.weights -= learning_rate * layer.dweights
        layer.biases -= learning_rate * layer.dbiases

    # print the loss/ cost every hundred loops
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


print("\nTraining complete!")
print("Final Predictions on training data:")
print(np.round(predictions, 3))

# Interactive testing loop
while True:
    try:
        # Get inputs
        a = float(input("\nEnter first input (0 or 1, or -1 to quit): "))
        if a == -1:
            break
        b = float(input("Enter second input (0 or 1): "))
        x_test = np.array([[a, b]])

        inputs = x_test
        for layer in layers:
            layer.forward(inputs)
            layer.output = sigmoid(layer.output)
            inputs = layer.output

        result = layers[-1].output[0][0]
        print(f"Prediction: {result:.4f} (interpreted as {'1' if result > 0.5 else '0'})")

    except KeyboardInterrupt:
        break






