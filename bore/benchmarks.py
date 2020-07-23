import numpy as np
import ConfigSpace

from hpbandster.core.worker import Worker

alpha = np.array([1.0, 1.2, 3.0, 3.2])
A = np.array([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
              [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
              [3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
              [17.0,  8.0,  0.05, 10.0,  0.1, 14.0]])
P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                     [2329, 4135, 8307, 3736, 1004, 9991],
                     [2348, 1451, 3522, 2883, 3047, 6650],
                     [4047, 8828, 8732, 5743, 1091,  381]])
dims = A.shape[1]


def hartmann(x):
    r = np.sum(A * np.square(x - P), axis=-1)
    return - np.dot(np.exp(-r), alpha)


class HartmannWorker(Worker):

    def compute(self, config, budget, **kwargs):

        X = np.hstack([config[f"x{i}"] for i in range(dims)])
        y = hartmann(X)

        return dict(loss=y, info=None)

    @staticmethod
    def get_configspace():
        cs = ConfigSpace.ConfigurationSpace()
        for i in range(dims):
            cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
                f"x{i}", lower=0, upper=1))
        return cs
