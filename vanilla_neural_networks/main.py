import typing
import time
import numpy as np

"""
To be implemented


Batch gradient descent (most imp, probably the only one I will be able to implement and test)

regularizers (L2, dropout)


Momentum, RMSprop, Adam optimizers

Batch Normalization? 
"""


def sparse_categorical_cross_entropy_loss(y_predicted, y_true):
    assert y_predicted.shape == y_true.shape, f"Found shape, y_predicted: {y_predicted.shape}, y_true:{y_true.shape}"

    n, m = y_true.shape #the output units, batch_size respectively

    return -np.sum(y_true * np.log(y_predicted+1e-9))/m


def softmax(z):

    temp = np.exp(z-np.max(z, axis=0, keepdims=True))
    return temp/np.sum(temp, axis=0, keepdims=True) #sum along columns
    #assuming z.shape = (units, batch_size)


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):

    return 1/(1+np.exp(-x))

def binary_cross_entropy_loss(y_predicted, y_true):
    return -np.mean((1-y_true)*(np.log(1-y_predicted+1e-9)) + y_true*np.log(y_predicted+1e-9))

def mean_squared_loss(y_predicted, y_true):
    return np.mean(np.square(y_predicted-y_true))

def accuracy(y_predicted, y_true):

    #input assumes y_pred.shape = (training examples, number of outputs)

    
    if (y_predicted.shape!=y_true.shape):
        raise IndexError(f"y_predicted.shape: {y_predicted.shape}, y_true: {y_true.shape}")
    
    # if (y_true.shape[1]==1):
    #     return np.mean(y_true==y_predicted, axis=1)
    
    return np.mean(np.all(y_true==y_predicted, axis=1))


def calculate_loss(A, y, loss_type):

    if loss_type == "binary_cross_entropy":
        return binary_cross_entropy_loss(A, y)
    elif (loss_type == "mse") or (loss_type == 'mean_squared_error'):
        return mean_squared_loss(A, y)
    elif (loss_type=="sparse_categorical_cross_entropy"):
        return sparse_categorical_cross_entropy_loss(A, y)
    else:
        raise NotImplementedError(f"Didn't implement loss type: {loss_type}")

        


#This was just defined because such a layer exists in tensorflow too
#currently there is no use for this in my codebase
class InputLayer:

    def __init__(self, units):
        self.units = units

    def forward_prop(self):
        pass

    def back_prop(self):
        pass

    def __call__(self, prev_layer):
        pass


class Dense:
    
    def __init__(self, units, activation='relu'):
        
        self.built=False #this is to track that the weights are not initialized
        self.units = units
        self.activation = activation
        self.W = None 
        self.b = None

    def build(self, input_shape):

        self.input_shape = input_shape
        # Used He/Xavier initialization (I don't remember the exact name)
        # Just to converge faster. could also give option for kernel_initializer
        # like TF does, but I believe that much customization is not need
        self.W = np.random.normal(0, np.sqrt(2/(input_shape+self.units)), (self.units, input_shape))
        self.b = np.zeros((self.units, 1))
        self.built=True

    def forward_prop(self, a_prev):

        z = np.matmul(self.W, a_prev) + self.b
        
        if (self.activation=='relu'):
            self.A = relu(z)
        elif self.activation=='sigmoid':
            self.A = sigmoid(z)
        elif self.activation=='softmax':
            self.A = softmax(z)        
        else:
            raise NotImplementedError("Only relu and sigmoid activation supported")    
        
        self.A_prev = a_prev
        self.Z = z
        
        return self.A
    
    def back_prop(self, dA):

        if (dA.shape[0] != self.units):
            raise Exception(f"value of dA.shape[0] not correct: expected {self.units}, found dA.shape={dA.shape}")
        
        
        if (self.activation=='relu'):
            dZ = dA * (self.Z>=0)
        elif (self.activation=='sigmoid'):
            # A = sigmoid(self.Z)
            dZ = dA * self.A*(1-self.A)
        elif self.activation=='softmax':
            dZ = dA  #assumes that the dA = A-Y is given
        else:
            raise Exception(f"Activation not supported: {self.activation}")
        
        self.db = np.mean(dZ, axis=1, keepdims=True) #take mean across all the batch sizes
        #keep dims is important because dim of bias is (n, 1) not (n,)

        m = dA.shape[1] #batch size basically
        self.dW = np.matmul(dZ, self.A_prev.T)/m

        dA_prev = np.matmul(self.W.T, dZ)

        return dA_prev
        



    def __call__(self, prev_layer):
        
        if not isinstance(prev_layer, Dense) and not isinstance(prev_layer, InputLayer):
            raise Exception("Layer has to be Dense or Input type")
        
        # this is currently not used anywhere throughout the code
        # but this can be used in back/forward propagation so as to not bother making a loop,
        # which is being currently done in .fit() function
        self.prev_layer = prev_layer


class Sequential:

    def __init__(self, layers:typing.List[typing.Union[Dense]]):
        self.layers=layers

    def compile(self, loss='mse'):
        # self.learning_rate = learning_rate
        # self.batch_size = batch_size
        # self.epochs=epochs
        self.loss = loss

        

    
    def build(self, input_size):
        """
        Initializes the weights and bias for each layer

        input_size: the number of units in the previous layer 
        """
        
        self.layers[0].build(input_size)
        self.layers[0].built=True

        for i in range(len(self.layers)-1):
            self.layers[i+1].build(self.layers[i].units)
            self.layers[i+1].built=True
    
    def fit(self, X, y, batch_size=32, epochs=10, learning_rate=1e-3, metrics=[], random_state=42):

        if (X.shape[0]!=y.shape[0]):
            raise ValueError(f"Training rows and labelled rows not same. X.shape[0]={X.shape[0]}, y.shape={y.shape}")

        #there is only 1 output for each input, assuming binary classficiation type
        # this has to be removed (or replaced by seperate validation if needed) when 
        # performing regression task, or multi-label or multi-class classficiation
        if (y.shape[1]!=1 and self.loss=='binary_crossentropy'):
            raise ValueError(f"expected y.shape[1] to be 1 but found { y.shape[1]}")
        
        # if the weights are not initialized, this initializes them
        if not self.layers[0].built:
            self.build(X.shape[1])
        
        

        # this is necessary because it is assumed that in input, each data point is arranged row-wise,
        # like in an excel sheet. But the neural network assumes that, number of rows = number of units,
        # and number of columns = batch_size
        X = X.T
        y = y.T

        N, M = X.shape #features, training examples

    


        for epoch_num in range(epochs):

            permutation = np.random.permutation(M)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            
            start = time.time()

            for batch_number in range(0, M, batch_size):
                x_batch = X_shuffled[:, batch_number:batch_number+batch_size]
                y_batch = y_shuffled[:, batch_number:batch_number+batch_size]


                A = x_batch
                for layer in self.layers:
                    A = layer.forward_prop(A)


                temp = np.zeros_like(A) #this is for measuing accuracy. predicting the class from final outputs
                temp[np.argmax(A, axis=0), np.arange(0, A.shape[1])] = True

                accuracy_ = accuracy((temp).T, y_batch.T)
            
                loss = calculate_loss(A, y_batch, self.loss)

                # print(f"Epoch: {i+1}, batch_number: {batch_number//batch_size+1}, Accuracy: {accuracy_}, {self.loss} loss: {loss}")
            



                if (self.loss == 'binary_cross_entropy'):
                    dA = (A-y_batch)/(A+1e-9)/(1-A+1e-9)
                elif (self.loss == "mse") or (self.loss == 'mean_squared_error'):
                    raise NotImplementedError
                    dA = 0 
                elif (self.loss=='sparse_categorical_cross_entropy'):
                    dA = A-y_batch
                else:
                    raise NotImplementedError(f"{self.loss} is not implemented yet")


                for i in range(len(self.layers)):
                    dA = self.layers[len(self.layers)-i-1].back_prop(dA)

                for layer in self.layers:
                    layer.W -= learning_rate*layer.dW
                    layer.b -= learning_rate*layer.db
            
            
            print(f"Epoch: {epoch_num+1}, time taken: {time.time()-start}")

            # A = self.predict(X.T).T
            # temp = np.zeros_like(A) #this is for measuing accuracy. predicting the class from final outputs
            # temp[np.argmax(A, axis=0), np.arange(0, A.shape[1])] = True
            # accuracy_ = accuracy((temp).T, y.T)
            # loss = calculate_loss(A, y, self.loss)
            
            # print(f"Epoch: {epoch_num+1}, Accuracy: {accuracy_}, {self.loss} loss: {loss}")

            
            
    def predict(self, X):

        #it is assumed as X.shape = (examples, number_of_features)

        X = X.T 
        for layer in self.layers:
            X = layer.forward_prop(X)

        return X.T #output (examples, number_of_output)
        


        








