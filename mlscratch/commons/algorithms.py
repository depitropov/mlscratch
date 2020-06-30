import numpy as np


class Minimization:

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


class Scaling:

    @staticmethod
    def normalize(array):
        array_max = np.max(array, axis=0)
        array_min = np.min(array, axis=0)
        array_max_min = array_max - array_min

        return (array - array_min) / array_max_min

    @staticmethod
    def add_identity_column(features):
        return np.c_[np.ones((features.shape[0], 1)), features]


