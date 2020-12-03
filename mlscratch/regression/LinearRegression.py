import numpy as np

from mlscratch.commons.algorithms import Scaling


class LinearRegression:

    def __init__(self, normalize=True, algorithm='gd'):
        self.parameters = None
        self.normalize = normalize

        if algorithm not in ('gd', 'norm'):
            raise Exception("Algorithms must be either 'gd' for GradientDescent or 'norm' for Normal equation")

        self.algorithm = algorithm

    def train(self, features, target, alpha=0.0005, iterations=2000):
        if self.algorithm == 'gd':
            self.parameters = self._vectorized_gradient_descent(features, target, alpha, iterations)
        elif self.algorithm == 'norm':
            self.parameters = self._normal_equation(features, target)

    def predict(self, features):
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        return LinearRegression._hypothesis(self.parameters, features)

    def _vectorized_gradient_descent(self, features, targets, alpha=0.0005, iterations=2000):
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        parameters = np.zeros((features.shape[1], 1))
        previous_error = 0

        for j in range(iterations):
            parameters = parameters - (alpha / features.shape[0] / 2) * (
                features.T.dot(LinearRegression._hypothesis(parameters, features) - targets)
            )

            if j % 100 == 0:
                error = LinearRegression._cost_function(parameters, features, targets)
                if abs(error - previous_error) < 0.0001:
                    return parameters
                previous_error = error
                print('>iteration=%d, error=%.3f' % (j, error))

        return parameters

    def _normal_equation(self, features, targets):
        if self.normalize:
            features = Scaling.normalize(features)
            targets = Scaling.normalize(targets)
        features = Scaling.add_identity_column(features)

        return np.linalg.pinv((features.T.dot(features))).dot(features.T.dot(targets))

    @staticmethod
    def _hypothesis(parameters, features):
        return features.dot(parameters)

    @staticmethod
    def _cost_function(parameters, features, targets):
        return np.sum((LinearRegression._hypothesis(parameters, features) - targets) ** 2) / features.shape[0]
