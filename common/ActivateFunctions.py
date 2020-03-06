'''
Created on 12 Feb 2019

@author: danhbuithi
'''
import numpy as np

def relu(z):
    return z * (z > 0)
    
def sigmoid(z):
    return 1/(1.0 + np.exp(-z))


def sigmoid1(z):
    if z >= 0:
        return 1.0/(1.0 + np.exp(-z))
    else:
        return np.exp(z)/(np.exp(z) + 1.0)

def tanh(z):
    return 2.0/(1 + np.exp(-2*z)) - 1.0

def detanh(z):
    return (1.0 - tanh(z)**2)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis = 0))
    return e_z/np.sum(e_z, axis = 0)
    
    
def dsigmoid(z):
    c = sigmoid(z)
    return c * (1 - c)
    
    
def drelu(z):
    return (z > 0).astype(int)
