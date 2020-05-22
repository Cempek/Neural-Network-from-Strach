import numpy as np

def sigmoid(z):
    output = 1 / (1 + np.exp(z))
    return output
def sigmoid_prime(z):
    return sigmoid(z) * sigmoid(1-z)

def tanh(z):
    return np.tanh(z)
def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(z,0)
def relu_prime(z):
    z[ z <= 0] = 0
    z[ z > 0] = 1
    return z

# This is the Neural Network designed for (input 2 neurons, hidden layer 3 neurons , output = 1)

class Network:
    def __init__(self):
        self.weight1 = np.random.rand(2,3)
        self.weight2 = np.random.rand(3,1)
        self.bias1 = np.random.rand(1)
        self.bias2 = np.random.rand(1)
   
    def train(self, X_train, y_train, epochs, learning_rate):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs        
        self.learning_rate = learning_rate
        losses = []
        for i in range(epochs):
            
            for x,y in zip(X_train, y_train):
                # Feed Forward
                hidden_layer = np.dot(x, self.weight1) + self.bias1
                activated_hidden_layer = tanh(hidden_layer)
                output= np.dot(activated_hidden_layer, self.weight2) + self.bias2
                predicted = tanh(output)
                # Loss Function
                current_error = np.power(( y - predicted),2)            
                losses.append(current_error)
                loss = np.mean(losses)                              # MSE (loss function)
                
                # BackBropogarion (Chain Rule)
                delta_output =  2 * ( predicted - y ) * tanh_prime(output)                    
                delta_hidden = (delta_output * self.weight2) * tanh_prime(hidden_layer).T
                # Update the weights and biases 
                self.weight2 -= self.learning_rate * np.dot(activated_hidden_layer.T, delta_output)
                self.bias2 -= self.learning_rate * sum(delta_output)
                self.weight1 -= self.learning_rate  *np.dot(x.T,delta_hidden.T)
                self.bias1 -= self.learning_rate * sum(delta_hidden)
                
            print(f'{i+1}/{epochs} epoch, loss: {loss}')

    def predict(self, x_test):
        self.x_test = x_test
        hidden_layer = np.dot(self.x_test, self.weight1) + self.bias1
        activated_hidden_layer = tanh(hidden_layer)
        output = np.dot(activated_hidden_layer, self.weight2) + self.bias2
        predicted = tanh(output)
        for x, y in zip(x_test,predicted):
            print(f'x : {x} , predicted : {y}')
        
# Training the dataset (XOR)      
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

NN = Network()
NN.train(x_train, y_train, epochs = 1500, learning_rate = 0.1)
     
# Test the the Neural Network

x_test = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])           
NN.predict(x_test)              
            
                