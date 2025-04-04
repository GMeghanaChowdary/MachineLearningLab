import numpy as np

# Input (features) and output (target)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# Normalization
X = X / np.amax(X, axis=0)
y = y / 100

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

# Hyperparameters
epoch = 7000
lr = 0.1
inputlayer_neurons = 2  # Number of neurons in input layer
hiddenlayer_neurons = 3  # Number of neurons in hidden layer
output_neurons = 1  # Number of neurons in output layer

# Initialize weights and biases
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for i in range(epoch):
    # Forward propagation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    
    # Backpropagation
    E0 = y - output  # Error at output layer
    outgrad = derivatives_sigmoid(output)  # Gradient at output layer
    d_output = E0 * outgrad  # Delta for output layer

    EH = d_output.dot(wout.T)  # Error at hidden layer
    hiddengrad = derivatives_sigmoid(hlayer_act)  # Gradient at hidden layer
    d_hiddenlayer = EH * hiddengrad  # Delta for hidden layer

    # Updating weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

# Output results
print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", output)
