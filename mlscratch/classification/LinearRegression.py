import numpy as np


class LinearRegression:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.parameters = None

    def train(self, features, target, alpha, iterations):
        self.parameters = self.gradient_descent(features, target, alpha, iterations)
        
    def predict(self, features):
        return features.dot(features)

    def calculate_error_bias(self, parameters, features, targets):
        return features.shape[0] / 2 * (sum(features.dot(parameters) - targets))

    def calculate_error(self, parameters, features, targets, parameter):
        result = 0
        for i in range(features.shape[0]):
            result = result + (features[i].dot(parameters) - targets[i]) * features[i][parameter]
        return features.shape[0] / 2 * result

    def gradient_descent(self, features, targets, alpha, iterations):
        features = np.c_[np.ones((features.shape[0], 1)), features]
        current_parameters = np.ones((features.shape[1], 1))
        temp_parameters = np.ones((features.shape[1], 1))

        for j in range(iterations):
            total_error = 0
            for parameter in range(current_parameters.shape[1] + 1):
                if parameter == 0:
                    error = self.calculate_error_bias(current_parameters, features, targets)
                    total_error = total_error + error
                    temp_parameters[parameter] = current_parameters[parameter] - alpha * error
                else:
                    error = self.calculate_error(current_parameters, features, targets, parameter)
                    total_error = total_error + error
                    temp_parameters[parameter] = current_parameters[parameter] - alpha * error
            current_parameters = temp_parameters
            print('>iteration=%d, error=%.3f' % (j, total_error))
        return current_parameters










