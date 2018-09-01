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

    def forward(self, Input):
        self.input = Input
        #print("Input:", Input, "   weights:", self.weights.shape)
        preactivations = Input @ self.weights + self.bias
        result = self.activation_function(preactivations)
        return result


class ConvLayer(object):
    """A convolutional layer with stride = 1 and max-pooling"""

    def __init__(self, n_x_out, n_y_out, f_size):
        #self.n_x = n_x_in
        #self.n_y = n_y_in
        self.n_x_out = n_x_out
        self.n_y_out = n_y_out
        self.f_size = f_size                  # filter size
        self.filter = np.random.rand(f_size, f_size)

    def forward(self, Input):
        self.input = Input
        n_x_in, n_y_in = Input.shape
        result = np.empty((n_x_in - self.f_size + 1, n_y_in - self.f_size + 1))

        for i in range(n_x_in - self.f_size + 1):
            for j in range(n_y_in - self.f_size + 1):
                region = Input[i : i + self.f_size, j : j + self.f_size]
                result[i,j] = np.sum(region * self.filter)

        result_pooled = np.empty((self.n_x_out, self.n_y_out))
        step_x = result.shape[0] / self.n_x_out
        step_y = result.shape[1]  /self.n_y_out
        for i in range(self.n_x_out):
            for j in range(self.n_y_out):
                region = result[round(i*step_x) : round((i+1)*step_x),
                                round(j*step_y) : round((j+1)*step_y)]
                result_pooled[i,j] = np.max(region)
        return result_pooled


class NN(object):

    def __init__(self, n_inputs, neuronCounts, activation_functions):
        layerCount = len(neuronCounts)
        self.n_inputs = n_inputs
        self.neuronCounts = neuronCounts
        self.layers = [None] * layerCount
        self.layers[0] = Layer(n_inputs, neuronCounts[0], activation_functions[0])
        for i in range(1, layerCount):
            self.layers[i] = Layer(self.neuronCounts[i-1], self.neuronCounts[i], activation_functions[i])

    def run_NN(self, Input):
        result = Input
        for layer in self.layers:
            result = layer.forward(result)
        #print(result)
        return result

    def printNN(self):
        for i in range(len(self.neuronCounts)):
            print('Layer ',i)
            #print(self.layers[i].layerType)
            print('weights:')
            print(self.layers[i].weights)
            print('bias:', self.layers[i].bias)

    def save_NN(self, savename):
        pickle.dump(self, open( savename , "wb" ))


if __name__ == '__main__':
    #network = NN(2, [2], ['ReLU'])
    #network.printNN()
    #result = network.run_NN([1, 1])
    #print("Result:", result)
    conv_layer = ConvLayer(5, 5, 3)
    Input = np.random.rand(10,10)
    output = conv_layer.forward(Input)
    print(output)

