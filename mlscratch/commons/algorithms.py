import numpy as np

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


