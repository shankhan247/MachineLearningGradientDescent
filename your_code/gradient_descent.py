import numpy as np
from your_code import HingeLoss, SquaredLoss
from your_code import L1Regularization, L2Regularization
#from your_code import accuracy, metrics
from .metrics import accuracy

class GradientDescent:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        N,d = features.shape
        
        w = np.zeros((d+1))
        for i in range(d+1):
            w[i] = np.random.uniform(-0.1,0.1)
        """
        w = np.zeros((d))
        for i in range(d):
            w[i] = np.random.uniform(-0.1,0.1)
        w = np.append(w,0.5)
        """
        bias = np.ones((N,1))
        features_new = np.append(features,bias,1)

        converged = 0
        count = 0
        loss_init = 0
        self.loss_array = []
        self.accuracy = []
        min_batch = 0
        max_batch = batch_size
        while count <= max_iter and converged == 0:
            if batch_size is None:
                gradient = self.loss.backward(features_new,w,targets)
                loss_new = self.loss.forward(features_new, w, targets)
                loss_diff = loss_new - loss_init
                loss_abs = np.absolute(loss_diff)
                self.loss_array.append(loss_new)

                if loss_abs < 0.0001:
                    converged = 1

                else:
                    step = self.learning_rate*gradient
                    conv = w - step
                    w = conv
                count = count + 1
                loss_init = loss_new
            
                self.model = w  
                curr_predict = self.predict(features) 
                acc = accuracy(targets,curr_predict)
                self.accuracy.append(acc)
            else:
                np.random.shuffle(features_new)
                batch_features = features_new[min_batch:max_batch]
                gradient = self.loss.backward(batch_features,w,targets)
                loss_new = self.loss.forward(batch_features, w, targets)
                self.loss_epoch = loss_new
                step = self.learning_rate*gradient
                conv = w - step
                w = conv
                count = count + batch_size
                self.model = w  
                curr_predict = self.predict(features) 
                acc = accuracy(targets,curr_predict)
                self.accuracy_epoch = acc
                min_batch = min_batch + batch_size
                max_batch = max_batch + batch_size
                if max_batch >= N:
                    count = max_iter + 1
                



    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        N,d = features.shape
        bias = np.ones((N,1))
        features = np.append(features,bias,1)
        predictions = np.zeros((N))
        for i in range(N):
            h = np.dot(self.model,features[i])
            if h > 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        
        return predictions

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        raise NotImplementedError()
