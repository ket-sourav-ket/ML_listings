"""
@uthor: sourav

A basic three layer feed forward neural network(MLP) utilising sigmoid activation units(use ReLU instead
of sigmoid units for better performance)

The model uses gradient descent to optimize the cost function and uses backpropagation to compute the partial
derivatives and Jacobians.

The computations are vectorized over the entire dataset to avoid programming loops as much as possible

"""

import numpy as np


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.
    Parameters
    ------------
    n_hidden : {int}Number of hidden units.

    l2 : {float} Lambda value for L2-regularization.

    epochs : {int} Number of passes over the training set.
    
    eta : {float} Learning rate.

    seed : {int} Random seed for initializing weights

    Attributes
    -----------
    *none*

    """
    def __init__(self, n_hidden=30,l2=0., epochs=100, eta=0.001, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta

    def _onehot(self, y, n_classes):
        """
        Encode labels into one-hot representation
        
        Parameters
        ------------
        y : array, dimension = [n_examples] Target values.
        n_classes : {int} Number of classes

        Returns
        -----------
        onehot : array, dimension = (n_examples, n_labels)
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        #Compute logistic function (sigmoid)
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        #Compute forward propagation step

        # step 1: net input of hidden layer
        # [n_examples, n_features] dot [n_features, n_hidden] -> [n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels] -> [n_examples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """
        Compute cost function.
        
        Parameters
        ----------
        y_enc : array, dimension = (n_examples, n_labels) one-hot encoded class labels.
        output : array, dimension = [n_examples, n_output_units] Activation of the output layer (forward propagation)
        Returns
        ---------
        cost : {float} Regularized cost
        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))            #program crashes on certain datasets with ZeroDivisionError error
        term2 = (1. - y_enc) * np.log(1. - output)   #due to numpy underflow when computing activation out of output layer
        cost = np.sum(term1 - term2) + L2_term       #using ReLU activation units mitigates this problem somewhat   
                                                     #but this issue stil remains unfortunately :(
        return cost

    def predict(self, X):                            #strictly requires two dimensional matrix as input
                                                     #or program crashes with index and mis-allignment errors
        """                                          
        Predict class labels

        Parameters
        -----------
        X : array, dimension = [n_examples, n_features] Input feature matrix of the data
        Returns:
        ----------
        y_pred : array, dimension = [n_examples] Predicted class labels(class label indicated with natural numbers 0...n)

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train):
        """ 
        Learn weights from training data.
        
        Parameters
        -----------
        X_train : array, dimension = [n_examples, n_features] feature matrix of training examples
        y_train : array, dimension = [n_examples] column vector of class labels(must be numerically encoded)
        Returns:
        ----------
        self

        *Note*: ideally SGD(with mini batches) should be used for training and also a validation dataset should be used to keep 
        track of the model performance after each epoch. I have not done any of that cause it's a lot of work :(
        The model still performs pretty well nonetheless 
        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        # Weight initialization:

        # weights for input_layer -> hidden_layer
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden_layer -> output_layer
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[0:])

                # Backpropagation:

                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[0:]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden] -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_features, n_examples] dot [n_examples, n_hidden] -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[0:].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_examples] dot [n_examples, n_classlabels] -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            
        return self