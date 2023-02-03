"""
@uthor:sourav

logistic regression classifier based closely on the adaline model,
using the sigmoid activation function instead of the identity activation
function.

The cost function is optimized using gradient descent(batch updates over the whole dataset)
The cost function is obtained by minimizing the negative log likelihood(since we want to
maximize the likelihood function)

"""
"""
Logistic Regression Classifier using gradient descent.
Parameters
------------
l_rate : Learning rate (between 0.0 and 1.0)
n_iter : Passes over the training dataset(number of epochs).
seed : Random number generator seed for random weight initialization.

Fields
-----------
w_ : Weights after fitting.
cost_ : Logistic cost function value in each epoch.
    
"""
import numpy as np

class LogisticRegressionGD(object):
    
    def __init__(self, l_rate=0.05, n_iter=100, seed=1):
        self.eta = l_rate
        self.n_iter = n_iter
        self.random_state = seed

        
    """
    Fit training data. 
    Parameters
    ----------
    X : n_example*n-features Training dataset matrix, where n_examples is the number of examples and
    n_features is the number of features.

    y : Target values column vector

    """    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):                #calculates net input Z from the weights
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):                  #defines the sigmoid activation function
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):                      #computes the prediction based on the unit step decision function
        return np.where(self.net_input(X) >= 0.0, 1, 0)