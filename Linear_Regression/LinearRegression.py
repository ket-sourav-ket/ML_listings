import numpy as np


"""
@uthor: sourav
Simple linear regression model based on iterative fitting through gradient descent

"""


class LinearRegressionGradDesc:
    def __init__(self, eta=0.01, n_iter=50, seed=1):

        """
        eta: learning rate for the gradient descent optimizer
        n_iter: maximum number for epochs
        seed: radom seed to make fitting process reproducible

        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = seed

    def fit(self, X, y):                      # optimize cost function(SSE) via gradient descent
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):          #loop that goes over the entire training dataset
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)
