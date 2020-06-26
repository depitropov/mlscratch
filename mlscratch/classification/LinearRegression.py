import numpy as np


class LinearRegression:

    def __init__(self, normalize=True):
        self.parameters = None
        self.normalize = normalize

    def train(self, features, target, alpha, iterations):
        self.parameters = self._gradient_descent(features, target, alpha, iterations)

    def predict(self, features):
        result = []
        if self.normalize:
            features = self._normalize(features)
        features = np.c_[np.ones((features.shape[0], 1)), features]
        for i in range(features.shape[0]):
            result.append(features[i].dot(self.parameters))
        return result

    def _calculate_error(self, parameters, features, targets):
        result = 0
        for i in range(features.shape[0]):
            result = result + (features[i].dot(parameters) - targets[i]) ** 2
        return result / features.shape[0]

    def _calculate_slope(self, parameters, features, targets, parameter):
        result = 0
        for i in range(features.shape[0]):
            result = result + (features[i].dot(parameters) - targets[i]) * features[i][parameter]
        return features.shape[0] / 2 * result

    def _normalize(self, array):
        array_max = np.max(array, axis=0)
        array_min = np.min(array, axis=0)
        array_max_min = array_max - array_min

        return (array - array_min) / array_max_min

    def _gradient_descent(self, features, targets, alpha, iterations):
        if self.normalize:
            features = self._normalize(features)
        features = np.c_[np.ones((features.shape[0], 1)), features]
        current_parameters = np.zeros((features.shape[1], 1))
        temp_parameters = np.zeros((features.shape[1], 1))
        previous_error = 0

        for j in range(iterations):
            total_error = 0
            for parameter in range(current_parameters.shape[0]):
                error = self._calculate_error(current_parameters, features, targets)
                slope = self._calculate_slope(current_parameters, features, targets, parameter)
                total_error = total_error + error
                temp_parameters[parameter] = current_parameters[parameter] - alpha * slope
            current_parameters = temp_parameters
            if abs(total_error - previous_error) < 0.0001:
                return current_parameters
            previous_error = total_error
            print('>iteration=%d,, error=%.3f' % (j, total_error))
        return current_parameters










