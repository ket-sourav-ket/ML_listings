"""
@uthor:sourav

A simple perceptron implementation (for classification purposes) using Rosenblatt's learning rule.The implementation
is based on the MCP artificial neuron model.This model only works for linearly separable data but fails to converge
if the dataset is not linearly seperable.A maximum limit on the number of iterations through the dataset is imposed
to allow the program to exit in case it is trained on not linearly seperable data

************************************************
|                                              |
| The implementation utilises the numpy module |
|                                              |
************************************************
"""

import numpy as np

"""
perceptron classifier object definition:

parameters
-----------

l_rate:Learning rate for model

n_iter:Passes over the training dataset(epoch)

seed:random number generator seed to make the results reproducible

Attributes
-----------

w_ : array of weights after fititng the model to the data

errors_ : a list of the number of misclassifications in each epoch

"""
class Perceptron(object):
    def __init__(self,l_rate=0.01,n_iter=50,seed=1):
        self.l_rate=l_rate
        self.n_iter=n_iter
        self.seed=seed

    def fit(self,X,Y):      # X is the (n_example*n-features) input or feature matrix and Y is the corresponding output or target column vector
        rgen= np.random.RandomState(self.seed)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size= 1 + X.shape[1])  #set weights initially to small random numbers drawn from normal distribution with 
        self.errors_ = []                                               #a standard deviation sigma=0.01
        for epoch in range(self.n_iter):
            errors=0
            for X_i,target in zip(X,Y):          #X_i denotes the i the example in the training dataset
                update= self.l_rate * (target - self.predict(X_i))
                self.w_[1:] += update * X_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]     # calculates net input Z by combining the weights and the input feature values

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0,1,-1)   #computes the class label using unit step function as the threshold function


        

