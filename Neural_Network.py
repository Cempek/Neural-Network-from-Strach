# -*- coding: utf-8 -*-
"""
Created on Wed May 20 00:54:21 2020

@author: Cem
"""

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

# This is basic Network designed for (input =4 neurons, hidden layer =3 , output = 1)

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
        losses2 = []
        for i in range(epochs):
            
            for x,y in zip(X_train, y_train):
                hidden_layer = np.dot(x, self.weight1) + self.bias1
                activated_hidden_layer = tanh(hidden_layer)
                output= np.dot(activated_hidden_layer, self.weight2) + self.bias2
                predicted = tanh(output)
                
                #current_error = y - predicted
                current_error = np.power(( y - predicted),2)
                losses.append(current_error)
                loss = np.mean(losses) 
                
                # current_error2 = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))     # turevi nasil hesaplaniyor bilmiyorum
                # losses2.append(current_error2)
                # loss2 = np.mean(losses2) 
                
                delta_output =  2 * ( predicted - y ) * tanh_prime(output)                     # buraya output girmemizin nedeni, sigmoid_prime outputu sigmoid edip turebini aliyor
                delta_hidden = (delta_output * self.weight2) * tanh_prime(hidden_layer).T

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
        
        
        
            
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])

y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


NN = Network()

NN.train(x_train, y_train, 500, 0.1)
     
x_test = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])           
NN.predict(x_test)              
            
                