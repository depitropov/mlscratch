import numpy as np

from mlscratch.commons.algorithms import Scaling
from mlscratch.commons.functions import Activation


class LogisticRegression:

    def __init__(self, normalize=True):
        self.parameters = None
        self.normalize = normalize

    def train(self, features, targets, alpha=0.0005, iterations=2000):
        self.parameters = self._vectorized_gradient_descent(features, targets, alpha, iterations)

    def predict(self, features, boundary=0.5):
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)

        decision_boundary = np.vectorize(lambda prob: 1 if prob >= boundary else 0)
        return decision_boundary(LogisticRegression._hypothesis(self.parameters, features)).flatten()

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
