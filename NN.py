import numpy as np 
import pickle

def sigmoid(z):
    return np.array(1/(1+np.exp(-z)))

def ReLU(z):
    z[np.where(z<0)] = 0
    return z
   
def load_NN( loadname):
    return pickle.load( open( loadname , "rb" ) )    

class Layer(object): 

    def __init__(self, neuronCount, neuronCountNext, layerType, activation_function): 
        self.neuronCount = neuronCount 
        if layerType == 'output':
            self.weights = np.nan
        else:
            self.weights = np.random.rand(neuronCountNext,neuronCount)
        self.layerType = layerType 
        self.activation_function = activation_function 
        
    
class NN(object): 

    def __init__(self, neuronCounts, activation_functions):
        layerCount = len(neuronCounts)
        self.neuronCounts = neuronCounts
        self.layers = np.empty([layerCount], dtype = object)
        self.layers[layerCount-1] = Layer(self.neuronCounts[layerCount-1],0,'output', activation_functions[layerCount-1])
        for i in range(layerCount-2,0,-1):
            self.layers[i] = Layer(self.neuronCounts[i],self.neuronCounts[i+1],'hidden',activation_functions[i])
        self.layers[0] = Layer(self.neuronCounts[0],self.neuronCounts[1],'input', activation_functions[0])

    def run_NN(self, Input):
        for index,layer in np.ndenumerate(self.layers):
            if layer.activation_function =='ReLU':
                if index[0] == 0:
                    Output = ReLU(Input.T)
                else:
                    Output = ReLU(Output)
            else:
                if index[0] == 0:
                    Output = sigmoid(Input.T)
                else:
                    Output = sigmoid(Output)
            if index[0] < len(self.layers)-1:
                Output =  layer.weights @ Output
            else:
                break
        return Output
        
    def printNN(self):
        for i in range(len(self.neuronCounts)):
            print('Layer ',i)
            print(self.layers[i].layerType)
            print('weights:')
            print(self.layers[i].weights)
            
    def save_NN(self, savename):
        pickle.dump(self, open( savename , "wb" ))
        
    
