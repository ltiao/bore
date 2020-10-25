import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

# TODO(LT): Extract framework agnostic core enginer code from
# `plugins.hpbandster` and place it here.
Evaluations = namedtuple('Evaluations', ["targets", "inputs"])


class Record:

    def __init__(self):
        self.rungs = {}

    @staticmethod
    def get_key(x):
        return tuple(x.tolist())

    def append(self, x, y, b):
        e = self.rungs.setdefault(b, Evaluations(targets=[], inputs=[]))
        e.inputs.append(x)
        e.targets.append(y)

    def num_rungs(self):
        return len(self.rungs)

    def budgets(self):
        return sorted(self.rungs.keys())

    def budget(self, t):
        budgets = self.budgets()
        return budgets[t]

    def _rung_size_from_budget(self, b):
        rung_size = len(self.rungs[b].targets)
        assert rung_size == len(self.rungs[b].inputs), \
            f"number of inputs and targets at budget `{b}` don't match!"
        return rung_size

    def rung_size(self, t):
        b = self.budget(t)
        return self._rung_size_from_budget(b)

    def rung_sizes(self):
        return [self._rung_size_from_budget(b) for b in self.budgets()]

    def size(self):
        return sum(self.rung_sizes())

    def rung_largest(self, min_size=0):
        t_largest = 0
        for t, b in enumerate(sorted(self.rungs)):
            if self._rung_size_from_budget(b) >= min_size:
                t_largest = t
        return t_largest

    def _threshold_from_budget(self, b, gamma):
        tau = np.quantile(self.rungs[b].targets, q=gamma)
        return tau

    def threshold(self, t, gamma):
        b = self.budget(t)
        return self._threshold_from_budget(b, gamma)

    def thresholds(self, gamma):
        return [self._threshold_from_budget(b, gamma) for b in self.budgets()]

    def _binary_labels_from_budget(self, b, gamma):
        tau = self._threshold_from_budget(self, b, gamma)
        return np.less(self.rungs[b].targets, tau)

    def binary_labels(self, t, gamma):
        b = self.budget(t)
        return self._binary_labels_from_budget(b, gamma)

    def sequences(self, gamma=None):
        sequences = {}
        for b in sorted(self.rungs):
            if gamma is not None:
                labels = self._binary_labels_from_budget(b, gamma)
            for x, y in zip(self.rungs[b].inputs, self.rungs[b].targets):
                key = self.get_key(x)
                ys = sequences.setdefault(key, [])
                ys.append(y)
        return sequences

    def foo(self, gamma=None):
        sequences = self.sequences(gamma=gamma)

        input_sequences = []
        target_sequences = []
        for x, ys in sequences.items():
            xb = np.expand_dims(x, axis=0)  # broadcast `x` before repeating
            input_sequence = np.repeat(xb, repeats=len(ys), axis=0)
            input_sequences.append(input_sequence)
            target_sequence = np.expand_dims(ys, axis=-1)
            target_sequences.append(target_sequence)
        return input_sequences, target_sequences

    def bar(self, gamma=None, pad_value=1e+20):
        input_sequences, target_sequences = self.foo(gamma=gamma)
        return (pad_sequences(input_sequences, dtype="float64",
                              padding="post", value=pad_value),
                pad_sequences(target_sequences, dtype="float64",
                              padding="post", value=pad_value))

    # def load_feature_matrix(self, t):
    #     return np.vstack(self.rungs[t].features)

    # def load_target_vector(self, t):
    #     return np.hstack(self.rungs[t].targets)

    # def load_regression_data(self, t):
    #     X = self.load_feature_matrix(t=t)
    #     y = self.load_target_vector(t=t)
    #     return X, y

    # def load_classification_data(self, gamma):
    #     X, y = self.load_regression_data()
    #     tau = np.quantile(y, q=gamma)
    #     z = np.less(y, tau)
    #     return X, z

    # def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
    #     # Clever ways of doing this would involve data structs. like KD-trees
    #     # or locality sensitive hashing (LSH), but these are premature
    #     # optimizations at this point, especially since the `any` below does lazy
    #     # evaluation, i.e. is early stopped as soon as anything returns `True`.
    #     return any(np.isclose(x_prev, x, rtol=rtol, atol=atol)
    #                for x_prev in self.features)
