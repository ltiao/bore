import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import Initializer
from sklearn.cluster import MiniBatchKMeans


class SubsetInitializer(Initializer):

    def __init__(self, inputs, seed=None):

        self.inputs = inputs
        self.num_inputs, *self.input_shape = inputs.shape

        self.seed = seed

    def __call__(self, shape, dtype=None):

        subset_size, *input_shape = shape
        assert self.input_shape == input_shape, "shape mismatch"

        # TODO: Make abstract base class or method that raises
        #   NotImplementedError
        # TODO: This just assumes `compute_subset` returns a numpy array.
        #   Would break if `dtype=tf.float64` for example.
        subset = self.compute_subset(subset_size, dtype)

        return tf.constant(subset, dtype=dtype)


class RandomSubset(SubsetInitializer):

    def __init__(self, inputs, seed=None):

        super(RandomSubset, self).__init__(inputs, seed=seed)
        self.random_state = np.random.RandomState(seed)

    def compute_subset(self, subset_size, dtype):

        ind = self.random_state.randint(self.num_inputs, size=subset_size)
        return self.inputs[ind]


class KMeans(SubsetInitializer):

    def compute_subset(self, subset_size, dtype):

        k_means = MiniBatchKMeans(subset_size,
                                  random_state=self.seed).fit(self.inputs)
        return k_means.cluster_centers_
