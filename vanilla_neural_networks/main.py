import typing
import numpy as np


def sigmoid(x):

    return 1/(1+np.exp(-x))

def binary_cross_entropy_loss(y_predicted, y_true):
    return -np.mean((1-y_true)*(np.log(1-y_predicted)) + y_true*np.log(y_predicted))

def mean_squared_loss(y_predicted, y_true):
    return np.mean(np.square(y_predicted-y_true))

def accuracy(y_predicted, y_true):
    return np.mean(y_predicted==y_true)

class InputLayer:

    def __init__(self, units):
        self.units = units

    def forward_prop(self):
        pass

    def back_prop(self):
        pass

    def __call__(self, prev_layer):
        
        if not isinstance(prev_layer, Dense):
            raise Exception("Layer has to be Dense")
        
        self.prev_layer = prev_layer


class Dense:
    
    def __init__(self, units, activation='relu'):
        
        
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None

    def build(self, input_shape):

        self.input_shape = input_shape
        self.W = np.random.normal(0, 2/(input_shape+self.units), (self.units, input_shape))
        self.b = 0

    def forward_prop(self, a_prev):

        z = np.matmul(self.W, a_prev) + self.b
        
        if (self.activation=='relu'):
            a = np.maximum(0, z)

        self.A = a
        self.A_prev = a_prev
        self.Z = z
        
        return a
    
    def back_prop(self, dA):

        if (dA.shape[0] != self.units):
            raise Exception(f"value of dA.shape[0] not correct: expected {self.units}, found {dA.shape[0]}")
        
        
        if (self.activation=='relu'):
            dZ = dA * (self.Z>=0)
        elif (self.activation=='sigmoid'):
            A = sigmoid(self.Z)
            dZ = dA * A*(1-A)
        else:
            raise Exception(f"Activation not supported: {self.activation}")
        
        self.db = np.mean(dZ, axis=1, keepdims=True)

        m = dA.shape[1] #batch size basically
        self.dW = np.matmul(dZ, self.A_prev.T)/m
        # self.dZ = dZ

        dA_prev = np.matmul(self.W.T, dZ)

        return dA_prev
        



    def __call__(self, prev_layer):
        
        if not isinstance(prev_layer, Dense) and not isinstance(prev_layer, InputLayer):
            raise Exception("Layer has to be Dense or Input type")
        
        self.prev_layer = prev_layer


class Sequential:

    def __init__(self, layers:typing.List[typing.Union[Dense]]):
        self.layers=layers

    def compile(self, loss='mse'):
        # self.learning_rate = learning_rate
        # self.batch_size = batch_size
        # self.epochs=epochs
        self.loss = loss
    
    def fit(self, X, y, batch_size=32, epochs=10, learning_rate=1e-3):

        if (X.shape[0]!=y.shape[0]):
            raise ValueError(f"Training rows and labelled rows not same. X.shape[0]={X.shape[0]}, y.shape={y.shape}")

        if (y.shape[1]!=1):
            raise ValueError(f"expected y.shape[1] to be 1 but found { y.shape[1]}")
        
        #currently ignoring batch_size 

        for i in range(epochs):

            A = X
            for layer in self.layers:
                A = layer.forward_prop(A)

            
            



            if (self.loss == 'binary_cross_entropy'):
                dA = (A-y)/A/(1-A)
            elif (self.loss == "mse") or (self.loss == 'mean_squared_error'):
                raise NotImplementedError
                dA = 0 

            for i in range(len(self.layers)):
                dA = self.layers[len(self.layers)-i-1].back_prop(dA)
            
            for layer in self.layers:
                layer.W -= learning_rate*layer.dW
                layer.b -= learning_rate*layer.db
            

        
        








