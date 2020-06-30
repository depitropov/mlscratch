import numpy as np

from mlscratch.commons.algorithms import Minimization, Scaling


class LogisticRegression:

    def __init__(self, normalize=True, algorithm='gd'):
        self.parameters = None
        self.normalize = normalize

        if algorithm not in ('gd', 'norm'):
            raise Exception("Algorithms must be either gd for GradientDescent or norm for Normal equation")

        self.algorithm = algorithm

    def train(self, features, target, alpha=0.0005, iterations=2000):
        if self.algorithm == 'gd':
            self.parameters = Minimization.gradient_descent(features, target, self._calculate_error,
                                                            self._calculate_slope, alpha, iterations)
        elif self.algorithm == 'norm':
            self.parameters = self._normal_equation(features, target)

    def predict(self, features):
        # TODO: Improve and check input
        result = []
        if self.normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        for i in range(features.shape[0]):
            result.append(features[i].dot(self.parameters))
        return result

    def _normal_equation(self, features, targets):
        if self.normalize:
            features = Scaling.normalize(features)
            targets = Scaling.normalize(targets)
        features = Scaling.add_identity_column(features)

        return np.linalg.pinv((features.T.dot(features))).dot(features.T.dot(targets))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

    @staticmethod
    def cost_function(parameters, features, targets):
        return 1

    @staticmethod
    def gradient_descent(features, targets, error_algorithm, slope_algorithm, alpha=0.0005, iterations=2000, normalize=True):
        if normalize:
            features = Scaling.normalize(features)
        features = Scaling.add_identity_column(features)
        current_parameters = np.zeros((features.shape[1], 1))
        temp_parameters = np.zeros((features.shape[1], 1))
        previous_error = 0

        for j in range(iterations):
            total_error = 0

            current_parameters = current_parameters - (alpha / features.shape[0]).dot(
                LogisticRegression.cost_function()
            )

            for parameter in range(current_parameters.shape[0]):
                if j % 100 == 0:
                    error = error_algorithm(current_parameters, features, targets)
                    total_error = total_error + error
                slope = slope_algorithm(current_parameters, features, targets, parameter)
                temp_parameters[parameter] = current_parameters[parameter] - alpha * slope
            current_parameters = temp_parameters
            if j % 100 == 0:
                if abs(total_error - previous_error) < 0.0001:
                    return current_parameters
                previous_error = total_error
                print('>iteration=%d,, error=%.3f' % (j, total_error))
        return current_parameters


    @staticmethod
    def _calculate_error(parameters, features, targets):
        result = 0
        for i in range(features.shape[0]):
            result = result + (features[i].dot(parameters) - targets[i]) ** 2
        return result / features.shape[0]

    @staticmethod
    def _calculate_slope(parameters, features, targets, parameter):
        result = 0
        for i in range(features.shape[0]):
            result = result + (features[i].dot(parameters) - targets[i]) * features[i][parameter]
        return features.shape[0] / 2 * result


