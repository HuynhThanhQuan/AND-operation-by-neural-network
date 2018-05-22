# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:35:00 2018

@author: HUYNH THANH QUAN
@email: hthquan28@gmail.com
"""

"""
Summary: AND Operation using Neural Network with 2 hidden layers

"""

import numpy as np

#Input
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#Desired value
y = np.array([[0], [0], [0], [1]])

#Random of numpy will generate a fix random numbers (remove to get different random numbers)
np.random.seed(1)

'''
This function calculate the sigmoid function with der = False
and derivative of sigmoid function in backpropagation with der = True
'''
def sigmoid(x, der = False):
    if der == True:
        return x * (1-x)
    else:
        return 1/(1+np.exp(-x))
    
'''
w0 is the weight matrix between the input layer to layer 1
w1 is the weight matrix between the layer 1 to layer 2
w2 is the weight matrix between the layer 2 to output layer
'''
w0 = np.random.random((2, 10))
w1 = np.random.random((10, 15))
w2 = np.random.random((15, 2))

'''
Iterate N times of this neural network, 
you can choose 10, 100, 1000 to see the differences in output layer
'''
for i in range(10000):
    
    #feed forward
    l0 = x                          #Input layer
    l1 = sigmoid(np.dot(l0, w0))    #Hidden layer 1
    l2 = sigmoid(np.dot(l1, w1))    #Hidden layer 2
    l3 = sigmoid(np.dot(l2, w2))    #Actual value by Neural network
    
    #backpropagation
    l3_err = y - l3                         #Error of desired value to actual value
    l3_del = l3_err * sigmoid(l3, True)     #Delta of layer 3 after pass activation function
    l2_err = np.dot(l3_del, w2.T)           #Error of layer 2 to layer 3
    l2_del = l2_err * sigmoid(l2, True)     
    l1_err = np.dot(l2_del, w1.T)
    l1_del = l1_err * sigmoid(l1, True)
    
    w2 += np.dot(l2.T, l3_del)              #Update weights 
    w1 += np.dot(l1.T, l2_del)
    w0 += np.dot(l0.T, l1_del)
    
print(y)
print(l3)