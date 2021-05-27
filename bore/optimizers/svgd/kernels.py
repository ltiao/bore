import numpy as np


class RadialBasis:

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def value_and_grad(self, X):
        length_scale = self.length_scale
        Z = X / length_scale
        diff = np.expand_dims(Z, axis=1) - Z
        sqdiff = np.square(diff)
        K = np.exp(-.5*np.sum(sqdiff, axis=-1))
        diff /= length_scale
        K_grad = np.sum(diff * np.expand_dims(K, axis=-1), axis=1)
        return K, K_grad
