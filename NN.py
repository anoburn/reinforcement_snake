import numpy as np
import pickle

def sigmoid(z):
    return np.array(1/(1+np.exp(-z)))

def sigmoid_diff(z):
    return sigmoid(z) * sigmoid(-z)

def ReLU(z):
    z[np.where(z<0)] = 0
    return z

def ReLU_diff(z):
    return np.where(z>0, 1, 0)

activation_functions = {'ReLU': ReLU, 'sigmoid': sigmoid}
derivative_functions = {'ReLU': ReLU_diff, 'sigmoid': sigmoid_diff}

def load_NN( loadname):
    return pickle.load( open( loadname , "rb" ) )


class Layer(object):

    def __init__(self, n_input, neuronCount, activation_function):
        self.neuronCount = neuronCount
        self.weights = np.random.rand(n_input, neuronCount)
        self.bias = np.random.rand(neuronCount)
        self.activation_function = activation_functions[activation_function]
        self.derivative_function = derivative_functions[activation_function]

    def forward(self, Input):
        self.input = Input
        self.preactivations = Input @ self.weights + self.bias
        result = self.activation_function(self.preactivations)
        return result

    def backward(self, upstream_gradient):

        # backprop through the activation function, which isn't modified
        activation_gradient = upstream_gradient * self.derivative_function(self.preactivations)

        self.grad_bias = activation_gradient
        self.grad_weights = np.outer(self.input, activation_gradient)
        downstream_gradient = activation_gradient @ self.weights.T
        return downstream_gradient

    def update(self, learning_rate):
        self.bias    -= learning_rate * self.grad_bias
        self.weights -= learning_rate * self.grad_weights


class ConvLayer(object):
    """A convolutional layer with stride = 1 and max-pooling"""

    def __init__(self, n_x_out, n_y_out, f_size):
        #self.n_x = n_x_in
        #self.n_y = n_y_in
        self.n_x_out = n_x_out
        self.n_y_out = n_y_out
        self.f = f_size                  # filter size
        self.filter = np.random.rand(f_size, f_size)

    def forward(self, Input):
        self.input = Input
        n_x_in, n_y_in = Input.shape
        result = np.empty(Input.shape)
        shift = int(self.f / 2)         # Amount of padding
        self.padding = lambda x: np.pad(x, shift, 'edge')   # pad input by repeating edges
        padded = self.padding(Input)
        for i in range(n_x_in):
            for j in range(n_y_in):
                region = padded[i : i + self.f, j : j + self.f]
                result[i,j] = np.sum(region * self.filter)

        result_pooled = np.empty((self.n_x_out, self.n_y_out))
        # Remember where the maximums are for backward step later
        self.max_locations = np.empty((2, self.n_x_out, self.n_y_out), dtype=int)

        # Sizes of sections from which the maximum is picked for pooling
        step_x = n_x_in / self.n_x_out
        step_y = n_y_in / self.n_y_out

        for i in range(self.n_x_out):
            for j in range(self.n_y_out):
                region = result[round(i*step_x) : round((i+1)*step_x),
                                round(j*step_y) : round((j+1)*step_y)]
                loc = np.unravel_index(np.argmax(region), region.shape)
                result_pooled[i,j] = region[loc]
                self.max_locations[0, i, j] = loc[0] + round(i*step_x)
                self.max_locations[1, i, j] = loc[1] + round(j*step_y)
        return result_pooled

    def backward(self, upstream_gradient):
        n_x_in, n_y_in = self.input.shape
        shift = int(self.f / 2)

        # Propagate gradient back through pooling, zero for every place not picked
        pool_gradient = np.zeros(self.input.shape)
        pool_gradient[self.max_locations[0], self.max_locations[1]] = upstream_gradient
        pool_gradient_padded = self.padding(pool_gradient)
        input_padded = self.padding(self.input)

        downstream_gradient = np.zeros((n_x_in+2*shift, n_y_in+2*shift))
        grad_filter = np.zeros((self.f, self.f))

        for i in range(n_x_in):
            for j in range(n_y_in):
                downstream_gradient[i:i+self.f, j:j+self.f] += self.filter * pool_gradient_padded[i,j]
                grad_filter += input_padded[i:i+self.f, j:j+self.f] * pool_gradient_padded[i,j]

        self.grad_filter = grad_filter
        return downstream_gradient[shift:-shift, shift:-shift]


class FlattenLayer(object):
    """A layer that flattens output from a convolutional layer to one dimension"""
    def __init__(self, input_shape, n_output):
        assert np.prod(input_shape) == n_output
        self.input_shape = input_shape
        self.n_output = n_output

    def forward(self, Input):
        #print("FlattenLayer forward: \nInput: (", Input.shape, ")", Input, "  expected shape:", self.input_shape)
        assert Input.shape == self.input_shape
        result = Input.flatten()
        return result

    def backward(self, upstream_gradient):
        downstream_gradient = upstream_gradient.reshape(self.input_shape)
        return downstream_gradient


class NN(object):

    def __init__(self, n_inputs, neuronCounts, activation_functions):
        layerCount = len(neuronCounts)
        self.n_inputs = n_inputs
        self.neuronCounts = neuronCounts
        self.layers = []
        index = 0
        if type(n_inputs) == int:
            self.layers.append(Layer(n_inputs, neuronCounts[0], activation_functions[0]))
            index += 1
        for i in range(index, layerCount):
            neuronCount = self.neuronCounts[i]
            if type(neuronCount) == int:
                neuronCountPrev = self.neuronCounts[i-1]
                if type(neuronCountPrev) != int:
                    neuronCountPrev = np.prod(neuronCountPrev[:-1])
                self.layers.append(Layer(neuronCountPrev, self.neuronCounts[i], activation_functions[i]))
            else:
                self.layers.append(ConvLayer(*neuronCount))
                neuronCountNext = self.neuronCounts[i+1]
                if type(neuronCountNext) == int:
                    #print("NeuronCount:", neuronCount, "  neuronCountNext:", neuronCountNext)
                    self.layers.append(FlattenLayer(neuronCount[:-1], neuronCountNext))

    def run(self, Input):
        result = Input
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backprop(self, gradient):
        downstream_gradient = gradient
        for layer in reversed(self.layers):
            downstream_gradient = layer.backward(downstream_gradient)
        return downstream_gradient

    def printNN(self):
        for i, layer in enumerate(self.layers):
            print('Layer ', i)
            if isinstance(layer, Layer):
                print('weights:')
                print(layer.weights)
                print('bias:', layer.bias)
            elif isinstance(layer, ConvLayer):
                print('filter:')
                print(layer.filter)

    def save_NN(self, savename):
        pickle.dump(self, open( savename , "wb" ))


if __name__ == '__main__':
    network = NN([10,10], [(3,3,3), 9, 4], [None, 'ReLU', 'ReLU'])
    #network.printNN()
    #result = network.run_NN([1, 1])
    #print("Result:", result)
    #conv_layer = ConvLayer(3, 3, 3)
    Input = np.random.rand(10,10)
    result = network.run(Input)
    print(result)
    error = np.ones(4)
    back = network.backprop(error)
    #output = conv_layer.forward(Input)
    #print(output)
    #flat = FlattenLayer((3,3), 9)
    #output = flat.forward(output)
    #print(output)
    #error = np.ones(9)
    #error_back = flat.backward(error)
    #print(error_back)
    #error = np.ones((3,3))
    #back = conv_layer.backward(error)


