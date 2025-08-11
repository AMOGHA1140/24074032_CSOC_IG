import typing
import time
import numpy as np

"""
This library does not make use of autograd.
"""

"""
To be implemented


Batch gradient descent (done)

regularizers (L2, dropout) - (done)


Momentum, RMSprop, Adam optimizers (done)

Batch Normalization? 
"""




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

def sparse_categorical_cross_entropy_loss(y_predicted, y_true):
    assert y_predicted.shape == y_true.shape, f"Found shape, y_predicted: {y_predicted.shape}, y_true:{y_true.shape}"
    n, m = y_true.shape #the output units, batch_size respectively
    
    return -np.sum(y_true * np.log(y_predicted+1e-9))/m

#these are ONLY supported for binary classification. their behaviour on multi class classficiation is undefined
def calculate_precision(y_predicted, y_true):

    assert y_predicted.shape==y_true.shape, f"expected shapes to be same, found, y_predicted.shape={y_predicted.shape}, y_true.shape={y_true.shape}"

    TP = np.sum((y_predicted==True) & (y_true==True))
    FP = np.sum((y_predicted==True) & (y_true==False))

    return (TP)/(TP+FP+1e-9)

def calculate_recall(y_predicted, y_true):

    assert y_predicted.shape==y_true.shape, f"expected shapes to be same, found, y_predicted.shape={y_predicted.shape}, y_true.shape={y_true.shape}"

    TP = np.sum((y_predicted==True) & (y_true==True))
    FN = np.sum((y_predicted==False) & (y_true==False))

    return (TP)/(TP+FN)

def calculate_f1(y_predicted, y_true):

    R = calculate_recall(y_predicted, y_true)
    P = calculate_precision(y_predicted, y_true)

    return 2*R*P/(R+P)

def accuracy(y_predicted, y_true):

    #input assumes y_pred.shape = (training examples, number of outputs)

    
    if (y_predicted.shape!=y_true.shape):
        raise IndexError(f"y_predicted.shape: {y_predicted.shape}, y_true: {y_true.shape}")
    

    
    if (y_true.shape[1]==1):
        return np.mean(y_true==y_predicted)
    
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



class Adam:

    def __init__(self, network, learning_rate=1e-3, beta1=0.9, beta2=0.999):
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.iteration=1
        self.learning_rate=learning_rate
        
        for layer in network.layers:
            layer.Sdw = np.zeros_like(layer.W)
            layer.Vdw = np.zeros_like(layer.W)
            layer.Sdb = np.zeros_like(layer.b)
            layer.Vdb = np.zeros_like(layer.b)

    def gradient_descent(self, network):

        for layer in network.layers:

            layer.Vdw = (self.beta1 * layer.Vdw + (1-self.beta1)*layer.dW)#/(1-self.beta1**self.iteration)
            layer.Vdb = (self.beta1 * layer.Vdb + (1-self.beta1)*layer.db)#/(1-self.beta1**self.iteration)

            layer.Sdw = (self.beta2 * layer.Sdw + (1-self.beta2)*np.square(layer.dW))#/(1-self.beta2**self.iteration)
            layer.Sdb = (self.beta2 * layer.Sdb + (1-self.beta2)*np.square(layer.db))#/(1-self.beta2**self.iteration)

            layer.W -= self.learning_rate* layer.Vdw / (np.sqrt(layer.Sdw) + 1e-8) * np.sqrt(1-self.beta2**self.iteration)/(1-self.beta1**self.iteration)
            layer.b -= self.learning_rate* layer.Vdb / (np.sqrt(layer.Sdb) + 1e-8) * np.sqrt(1-self.beta2**self.iteration)/(1-self.beta1**self.iteration)

        self.iteration+=1

class Momentum:

    def __init__(self, network, learning_rate=1e-3, beta=0.9):

        if beta>=1:
            raise ValueError(f"beta must be less than 1, found beta={beta}")

        self.beta = beta
        self.iteration = 1
        self.learning_rate = learning_rate
        for layer in network.layers:
            layer.Vdw = np.zeros_like(layer.W)
            layer.Vdb = np.zeros_like(layer.b)
    
    def gradient_descent(self, network):

        for layer in network.layers:
            layer.Vdw  = (self.beta*layer.Vdw + (1-self.beta)*layer.dW)#/(1-self.beta**self.iteration)
            layer.Vdb  = (self.beta*layer.Vdb + (1-self.beta)*layer.db)#/(1-self.beta**self.iteration)
            
            layer.W -= self.learning_rate*layer.Vdw
            layer.b -= self.learning_rate*layer.Vdb
        
        # self.iteration += 1

class SGD:

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def gradient_descent(self, network):

        for layer in network.layers:
            layer.W -= self.learning_rate*layer.dW
            layer.b -= self.learning_rate * layer.db


class L2Regularizer:

    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
    
    def calculate_loss(self, layer):

        return 0.5 * self.lambda_ * np.sum(np.square(layer.W))
    
    def calculate_gradient(self, layer):

        return self.lambda_ * layer.W
        


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


class Dropout:

    def __init__(self, keep_prob):
        if (keep_prob >= 1.0):
            raise ValueError(f"keep_prob must be strictly less than 1, got {keep_prob}")
        
        self.regularizer = None
        self.activation = None
        self.W = self.b = 0
        self.keep_prob = keep_prob

    def forward_prop(self, a_prev, is_training=False):

        if (not is_training):
            return a_prev

        self.D = (np.random.rand(*a_prev.shape) < self.keep_prob).astype(float)
        self.A = self.D * a_prev / self.keep_prob
        return self.A
    
    def back_prop(self, dA):

        self.dW = self.db = 0

        dA_prev = dA*self.D / self.keep_prob

        return dA_prev
    
    def build(self, input_shape):
        self.units = self.prev_layer.units
        self.input_shape = input_shape






class Dense:
    
    def __init__(self, units, activation='relu', regularizer=None):
        
        self.built=False #this is to track that the weights are not initialized
        self.units = units
        self.regularizer=regularizer
        
        if activation not in ['relu', 'softmax', 'sigmoid']:
            raise NotImplementedError(f"Activation type: {activation} not implemented")

        self.activation = activation
        self.W = None 
        self.b = None

    def build(self, input_shape):

        self.input_shape = input_shape
        # Used He/Xavier initialization (I don't remember the exact name)
        # Just to converge faster. could also give option for kernel_initializer
        # like TF does, but I believe that much customization is not needed at start
        self.W = np.random.normal(0, np.sqrt(2/(input_shape+self.units)), (self.units, input_shape))
        self.b = np.zeros((self.units, 1))
        self.built=True

    def forward_prop(self, a_prev, is_training=False):

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

        if (self.regularizer is not None):
            self.dW += self.regularizer.calculate_gradient(self)

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
        self.loss = None

    def compile(self, loss, optimizer=SGD()):
        # self.learning_rate = learning_rate
        # self.batch_size = batch_size
        # self.epochs=epochs
        self.loss = loss
        self.optimizer = optimizer

    def add_layer(self, layer):
        self.layers.append(layer)

    
    def build(self, input_size):
        """
        Initializes the weights and bias for each layer

        input_size: the number of units in the previous layer 
        """
        
        self.layers[0].build(input_size)
        self.layers[0].built=True

        for i in range(len(self.layers)-1):
            self.layers[i+1].prev_layer = self.layers[i]
            self.layers[i+1].build(self.layers[i].units)
            self.layers[i+1].built=True
    
    def fit(self, X, y, batch_size=32, epochs=10, metrics=[], random_state=None, X_test=None, y_test=None):

        if (X.shape[0]!=y.shape[0]):
            raise ValueError(f"Training rows and labelled rows not same. X.shape[0]={X.shape[0]}, y.shape={y.shape}")        
        if (y.shape[1]!=1 and self.loss=='binary_crossentropy'):
            raise ValueError(f"expected y.shape[1] to be 1 but found { y.shape[1]}")
        
        if (X_test is None)^(y_test is None):
            raise ValueError(f"{'X_test' if X_test is None else 'y_test'} is None while the other is provided")
        if (X_test is not None and y_test is not None) and (X_test.shape[0]!=y_test.shape[0]):
            raise ValueError(f"number of rows in X_test and y_test not same. X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        

        if random_state is not None:
            np.random.set_state(random_state)        
        # if the weights are not initialized, this initializes them
        if not self.layers[0].built:
            self.build(X.shape[1])
        
        

        # this is necessary because it is assumed that in input, each data point is arranged row-wise,
        # like in an excel sheet. But the neural network assumes that, number of rows = number of units,
        # and number of columns = batch_size
        X = X.T
        y = y.T
        
        if (X_test is not None): #take transpose of test set, similar to training set
            X_test = X_test.T
            y_test=y_test.T        


        N, M = X.shape #features, training examples

        test_accuracy = []
        train_accuracy = []
        test_loss = []
        train_loss = []

        

        for epoch_num in range(epochs):

            permutation = np.random.permutation(M)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            
            start = time.time()
            epoch_training_loss_=[]
            epoch_training_accuracy_ = []

            for batch_number in range(0, M, batch_size):
                x_batch = X_shuffled[:, batch_number:batch_number+batch_size]
                y_batch = y_shuffled[:, batch_number:batch_number+batch_size]

                

                A = x_batch
                for layer in self.layers:
                    A = layer.forward_prop(A, True)

                temp=0
                if (self.loss=='binary_cross_entropy'):
                    temp = A>0.5
                else:
                    temp = np.zeros_like(A) #this is for measuing accuracy. predicting the class from final outputs
                    temp[np.argmax(A, axis=0), np.arange(0, A.shape[1])] = True

                accuracy_ = accuracy((temp).T, y_batch.T)
                loss = calculate_loss(A, y_batch, self.loss)
                for layer in self.layers:
                    if layer.regularizer is not None:
                        loss += layer.regularizer.calculate_loss(layer)

                epoch_training_accuracy_.append(accuracy_)
                epoch_training_loss_.append(loss)

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

                self.optimizer.gradient_descent(self)

                # for layer in self.layers:
                #     layer.W -= learning_rate*layer.dW
                #     layer.b -= learning_rate*layer.db
            
            train_loss.append(np.mean(epoch_training_loss_))
            train_accuracy.append(np.mean(epoch_training_accuracy_))

            print(f"Epoch: {epoch_num+1}, time taken: {time.time()-start}, accuracy={train_accuracy[-1]}, {self.loss} loss={train_loss[-1]}")



            if X_test is not None:
                A = self.predict(X_test.T).T
                temp=0

                if (self.loss=='binary_cross_entropy'):
                    temp = A>0.5
                else:
                    temp = np.zeros_like(A) #this is for measuring accuracy. predicting the class from final outputs
                    temp[np.argmax(A, axis=0), np.arange(0, A.shape[1])] = True #this sets the max value in each example to true, others to false
                accuracy_ = accuracy((temp).T, y_test.T) 
                loss = calculate_loss(A, y_test, self.loss)

                test_accuracy.append(accuracy_)
                test_loss.append(loss)

                # print(f", {self.loss} loss: {loss}, Accuracy: {accuracy_}")
            
            
        return (train_loss, train_accuracy), (test_loss, test_accuracy)

            
            
    def predict(self, X):

        #it is assumed as X.shape = (examples, number_of_features)

        X = X.T 
        for layer in self.layers:
            X = layer.forward_prop(X)

        return X.T #output (examples, number_of_output)
        


