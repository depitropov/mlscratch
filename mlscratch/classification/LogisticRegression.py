import numpy as np

from mlscratch.commons.algorithms import Minimization, Scaling
from mlscratch.commons.functions import Activation


class LogisticRegression:

    def __init__(self, normalize=True):
        self.parameters = None
        self.normalize = normalize

    def train(self, features, targets, alpha=0.0005, iterations=2000):
        self.parameters = self._vectorized_gradient_descent(features, targets, alpha, iterations)

    def train2(self, features, targets, alpha, iterations):
        if self.normalize:
            features = Scaling.normalize(features)
        cost_history = []
        features = Scaling.add_identity_column(features)
        weights = np.zeros((features.shape[1], 1))

        for i in range(iterations):
            weights = LogisticRegression.update_weights(features, targets, weights, alpha)

            # Calculate error for auditing purposes
            cost = LogisticRegression._cost_function2(weights, features, targets)
            cost_history.append(cost)

            # Log Progress
            if i % 1000 == 0:
                print("iter: " + str(i) + " cost: " + str(cost))

        self.parameters = weights

    def predict(self, features):
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        return LogisticRegression._hypothesis(self.parameters, features)

    @staticmethod
    def predict2(weights, features):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        z = np.dot(features, weights)
        return Activation.sigmoid(z)

    def predict3(self, features):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        z = np.dot(features, self.parameters)
        return Activation.sigmoid(z)

    def _vectorized_gradient_descent(self, features, targets, alpha=0.0005, iterations=2000):
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        parameters = np.zeros((features.shape[1], 1))
        previous_error = 0

        for j in range(iterations):
            parameters = parameters - (alpha / features.shape[0]) * (
                features.T.dot(LogisticRegression._hypothesis(parameters, features) - targets)
            )

            if j % 100 == 0:
                error = LogisticRegression._cost_function(parameters, features, targets)
                if abs(error - previous_error) < 0.0001:
                    return parameters
                previous_error = error
                print('>iteration=%d, error=%.3f' % (j, error))

        return parameters

    @staticmethod
    def _hypothesis(parameters, features):
        return Activation.sigmoid(features.dot(parameters))

    @staticmethod
    def _cost_function(parameters, features, targets):
        return np.sum(-targets * (np.log(LogisticRegression._hypothesis(parameters, features)) - (1 - targets) * (
            np.log(1 - LogisticRegression._hypothesis(parameters, features))))) / features.shape[0]

    @staticmethod
    def _cost_function2(weights, features, labels):
        '''
        Using Mean Absolute Error

        Features:(100,3)
        Labels: (100,1)
        Weights:(3,1)
        Returns 1D matrix of predictions
        Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
        '''
        observations = len(labels)

        predictions = LogisticRegression.predict2(weights, features)

        # Take the error when label=1
        class1_cost = -labels * np.log(predictions)

        # Take the error when label=0
        class2_cost = (1 - labels) * np.log(1 - predictions)

        # Take the sum of both costs
        cost = class1_cost - class2_cost

        # Take the average cost
        cost = cost.sum() / observations

        return cost

    @staticmethod
    def update_weights(features, labels, weights, lr):
        '''
        Vectorized Gradient Descent

        Features:(200, 3)
        Labels: (200, 1)
        Weights:(3, 1)
        '''
        N = len(features)

        #1 - Get Predictions
        predictions = LogisticRegression.predict2(weights, features)

        #2 Transpose features from (200, 3) to (3, 200)
        # So we can multiply w the (200,1)  cost matrix.
        # Returns a (3,1) matrix holding 3 partial derivatives --
        # one for each feature -- representing the aggregate
        # slope of the cost function across all observations
        gradient = np.dot(features.T,  predictions - labels)

        #3 Take the average cost derivative for each feature
        gradient /= N

        #4 - Multiply the gradient by our learning rate
        gradient *= lr

        #5 - Subtract from our weights to minimize cost
        weights -= gradient

        return weights