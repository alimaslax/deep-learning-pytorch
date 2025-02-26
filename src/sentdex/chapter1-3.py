inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


# len weights, len biases == # of nuerons
# Output of current layer
layer_outputs = []
# For each neuron
#====================
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron’s output variable
        neuron_output += n_input*weight
    # Add bias
    neuron_output += neuron_bias
    # Put neuron’s result to the layer’s output list
    layer_outputs.append(neuron_output)
print(layer_outputs)


# pip install ../libs/nns-0.5.1.tar.gz
# pip install matplotlib
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples= 100, classes=3)
plt.scatter(X[:,0], X[:,1],c= y, cmap= 'brg')
plt.show()


nnfs.init()
n_inputs = 2
n_neurons = 4
weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))
print(weights)
print(biases)


# Create dataset
X, y = spiral_data(samples=100, classes= 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print(dense1.output[:5])