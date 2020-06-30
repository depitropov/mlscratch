import numpy as np


class Activation:

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))
