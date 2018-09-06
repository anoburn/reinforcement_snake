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

def identity(z):
    return z

def identity_diff(z):
    return np.ones(z.shape)

activation_functions = {'ReLU': ReLU, 'sigmoid': sigmoid, 'identity': identity}
derivative_functions = {'ReLU': ReLU_diff, 'sigmoid': sigmoid_diff, 'identity': identity_diff}

def load_NN( loadname):
    return pickle.load( open( loadname , "rb" ) )


class Layer(object):

    def __init__(self, n_input, neuronCount, activation_function):
        self.neuronCount = neuronCount
        self.weights = np.random.rand(n_input, neuronCount)
        #self.weights = np.ones((n_input, neuronCount))
        self.bias = np.random.rand(neuronCount)
        #self.bias = np.zeros(neuronCount)
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
        #print("self.input:", self.input)
        #print("activ. grad:", activation_gradient)
        self.grad_weights = np.outer(self.input, activation_gradient)
        downstream_gradient = activation_gradient @ self.weights.T
        return downstream_gradient

    def update(self, learning_rate):
        #print("Grad_bias:", self.grad_bias)
        #print("Grad_weights:", self.grad_weights)
        self.bias    -= learning_rate * self.grad_bias
        self.weights -= learning_rate * self.grad_weights


class ConvLayer(object):
    """A convolutional layer with stride = 1 and max-pooling"""

    def __init__(self, n_x_out, n_y_out, f_size, dim_in, dim_out, activation_function):
        self.n_x_out = n_x_out      # Size of output pictures. Input size isn't needed, since filter works on any size
        self.n_y_out = n_y_out
        self.f = f_size             # filter size
        self.dim_in  = dim_in       # number of dimensions at input. This is needed for shape of filter
        self.dim_out = dim_out
        self.filter = np.random.rand(f_size, f_size, dim_in, dim_out)
        self.activation_function = activation_functions[activation_function]
        self.derivative_function = derivative_functions[activation_function]

    def forward(self, Input):
        """Performs a forward step using Input with shape (n_x_in, n_y_in, dim_in)"""
        self.input = Input
        n_x_in, n_y_in = Input.shape[:2]
        result = np.empty((*Input.shape, self.dim_out))
        shift = int(self.f / 2)         # Amount of padding
        self.edges = ((shift, shift), (shift, shift), (0,0))     # only pad picture space, not dimensions
        padding = lambda x: np.pad(x, self.edges, 'edge')   # pad input by repeating edges
        padded = padding(Input)
        # Convolve the input pictures with the corresponding dimension of filter.
        # The picture size is not changed yet
        for i in range(n_x_in):
            for j in range(n_y_in):
                region = padded[i : i + self.f, j : j + self.f]     # multidimensional are of shape (f, f, dim_in)
                # for each output dimension multiply corresponding part of filter for that output-dimension (f, f, dim_in) with the region
                for d in range(self.dim_out):
                    result[i,j,d] = np.sum(region * self.filter[:,:,:,d])

        result_pooled = np.empty((self.n_x_out, self.n_y_out, self.dim_out))
        # Remember where the maximums are for backward step later
        self.max_locations = np.empty((3, self.n_x_out, self.n_y_out), dtype=int)

        # Sizes of sections from which the maximum is picked for pooling
        step_x = n_x_in / self.n_x_out
        step_y = n_y_in / self.n_y_out

        for i in range(self.n_x_out):
            for j in range(self.n_y_out):
                for d in range(self.dim_out):
                    # region from which maximum is picked. Values can't be in several regions
                    # shape: (step_x, step_y)
                    region = result[round(i*step_x) : round((i+1)*step_x),
                                    round(j*step_y) : round((j+1)*step_y), d]
                    loc = np.unravel_index(np.argmax(region), region.shape)
                    result_pooled[i,j,d] = region[loc]
                    self.max_locations[0, i, j] = loc[0] + round(i*step_x)
                    self.max_locations[1, i, j] = loc[1] + round(j*step_y)
                    self.max_locations[2, i, j] = d
        self.preactivations = result_pooled
        result = self.activation_function(result_pooled)
        return result


    def backward(self, upstream_gradient):
        """Performs a backpropagation step using a gradient with shape (n_x_out, n_y_out, dim_out)"""
        n_x_in, n_y_in = self.input.shape[:2]
        # same as in forward()
        shift = int(self.f / 2)
        edges = self.edges
        padding = lambda x: np.pad(x, edges, 'edge')

        # backprop through the activation function, which isn't modified
        activation_gradient = upstream_gradient * self.derivative_function(self.preactivations)

        # Propagate gradient back through pooling, zero for every place not picked.
        pool_gradient = np.zeros((n_x_in, n_y_in, self.dim_out))
        for d in range(self.dim_out):
            pool_gradient[self.max_locations[0], self.max_locations[1], self.max_locations[2]] += activation_gradient[:,:,d]
        pool_gradient_padded = padding(pool_gradient)   # pad gradient for easier computations. Edges will be removed before returning
        input_padded = padding(self.input)

        downstream_gradient = np.zeros((n_x_in+2*shift, n_y_in+2*shift, self.dim_in))    # padded downstream gradient
        grad_filter = np.zeros((self.f, self.f, self.dim_in, self.dim_out))     # gradient of filter. This will be used to update it

        # Calculate downstream-/filter-gradient for each location and dimension
        for i in range(n_x_in):
            for j in range(n_y_in):
                for d_in in range(self.dim_in):
                    for d_out in range(self.dim_out):
                        downstream_gradient[i:i+self.f, j:j+self.f, d_in] += self.filter[:,:,d_in,d_out] * pool_gradient_padded[i,j,d_out]
                        grad_filter[:,:,d_in,d_out] += input_padded[i:i+self.f, j:j+self.f, d_in] * pool_gradient_padded[i,j,d_out]
        self.grad_filter = grad_filter
        return downstream_gradient[shift:-shift, shift:-shift]

    def update(self, learning_rate):
        self.filter -= learning_rate * self.grad_filter
        return


class FlattenLayer(object):
    """A layer that flattens output from a convolutional layer to one dimension"""
    def __init__(self, input_info, n_output):
        self.input_shape = (*input_info[:2], input_info[4])    # shape: (n_x, n_y, dim)
        assert np.prod(self.input_shape) == n_output
        self.n_output = n_output

    def forward(self, Input):
        #print("FlattenLayer forward: \nInput: (", Input.shape, ")", Input, "  expected shape:", self.input_shape)
        assert Input.shape == self.input_shape
        result = Input.flatten()
        return result

    def backward(self, upstream_gradient):
        downstream_gradient = upstream_gradient.reshape(self.input_shape)
        return downstream_gradient

    def update(self, learning_rate):
        pass


class NN(object):

    def __init__(self, n_inputs, neuronCounts, activation_functions):
        layerCount = len(neuronCounts)
        self.n_inputs = n_inputs
        self.neuronCounts = neuronCounts
        self.layers = []
        index = 0   # holds information to skip first layer in loop if it was added
        # Add first layer manually if it is a regular one, since it needs n_inputs
        if type(n_inputs) == int:
            self.layers.append(Layer(n_inputs, neuronCounts[0], activation_functions[0]))
            index += 1

        for i in range(index, layerCount):
            neuronCount = self.neuronCounts[i]

            # Regular neuron layer
            if type(neuronCount) == int:
                neuronCountPrev = self.neuronCounts[i-1]
                if type(neuronCountPrev) != int:
                    neuronCountPrev = np.prod(neuronCountPrev[:2]) * neuronCountPrev[4]
                self.layers.append(Layer(neuronCountPrev, self.neuronCounts[i], activation_functions[i]))

            # Convolutional layer. Adds flattening layer if next layer is regular
            else:
                self.layers.append(ConvLayer(*neuronCount, activation_functions[i]))
                neuronCountNext = self.neuronCounts[i+1]
                if type(neuronCountNext) == int:
                    #print("NeuronCount:", neuronCount, "  neuronCountNext:", neuronCountNext)
                    self.layers.append(FlattenLayer(neuronCount, neuronCountNext))

    def run(self, Input):
        result = Input
        #print("input:", Input)
        for layer in self.layers:
            result = layer.forward(result)
            #print("result:", result)
        return result

    def backprop(self, gradient):
        downstream_gradient = gradient
        for layer in reversed(self.layers):
            #print(downstream_gradient)
            downstream_gradient = layer.backward(downstream_gradient)
        return downstream_gradient

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)


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
    network = NN([10,10], [(3,3,3,3,2), 18, 4], [None, 'ReLU', 'ReLU'])
    #network = NN(6, [10, 4], ['ReLU', 'ReLU'])
    #network.printNN()
    #Input = np.ones(6)
    #network.run(Input)
    #error = np.array([0, 0, 0, 1])
    #network.backprop(error)
    #network.update(1)
    #network.backprop(error)
    #network.update(1)
    #network.printNN()
    #network.printNN()
    #result = network.run_NN([1, 1])
    #print("Result:", result)
    #conv_layer = ConvLayer(3, 3, 3)
    Input = np.random.rand(10,10,3)
    result = network.run(Input)
    print(result.shape)
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


