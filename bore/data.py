import numpy as np


class Record:

    def __init__(self):
        self.features = []
        self.targets = []
        self.budgets = []

    def size(self):
        return len(self.targets)

    def append(self, x, y, b=None):
        self.features.append(x)
        self.targets.append(y)
        if b is not None:
            self.budgets.append(b)

    def load_feature_matrix(self):
        return np.vstack(self.features)

    def load_target_vector(self):
        return np.hstack(self.targets)

    def load_regression_data(self):
        X = self.load_feature_matrix()
        y = self.load_target_vector()
        return X, y

    def load_classification_data(self, gamma):
        X, y = self.load_regression_data()
        tau = np.quantile(y, q=gamma)
        z = np.less(y, tau)
        return X, z

    # def to_dataframe(self):
    #     frame = pd.DataFrame(data=self.features).assign(budget=self.budgets,
    #                                                     loss=self.targets)
    #     return frame

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        # Clever ways of doing this would involve data structs. like KD-trees
        # or locality sensitive hashing (LSH), but these are premature
        # optimizations at this point, especially since the `any` below does lazy
        # evaluation, i.e. is early stopped as soon as anything returns `True`.
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol)
                   for x_prev in self.features)


class MultiFidelityRecord:

    def __init__(self, gamma=None):
        self._data = {}  # TODO(LT): Find more descriptive name
        self._targets = {}
        self.gamma = gamma

    @staticmethod
    def compute_key(x):
        return tuple(x.tolist())

    def append(self, x, y, b):

        k = self.compute_key(x)
        dct = self._data.setdefault(k, {})
        dct[b] = y

        # Note that there is no information contained in `self._targets`
        # that we couldn't derive from `self._data` but we are using this
        # additional data structure for time-efficiency albeit at the expense
        # of memory-efficiency.
        ys = self._targets.setdefault(b, [])
        ys.append(y)

        # TODO(LT): Decide how best to handle situation where y has already
        #   been recorded for a given (x, b)
        # if b in d and d[b] != y:
        #     print(f"target value for x={x} at budget b={b} already recorded! "
        #           f"Replacing... (old={d[b]:.3f}, new={y:.3f})")

    def num_rungs(self):
        """
        Get the total number of rungs recorded.

        Returns
        -------
        int
            Total number of rungs recorded.
        """
        return len(self._targets)

    def highest_rung(self, min_size=1):
        """
        Get the highest rung attained so far that has recorded at least some
        given number of evaluations.

        Parameters
        ----------
        min_size : int, optional
            Minimum number of evaluations observed at a rung (default: 1).
        Returns
        -------
        int
            The highest rung with at least ``min_size`` observations.
        """
        # Equivalent to:
        # max(filter(lambda u: self._rung_size_from_budget(u[1]) >= min_size,
        #            enumerate(sorted(self.rungs))), default=None)
        t_max = None
        for t, b in enumerate(self.budgets()):
            if self._rung_size_from_budget(b) >= min_size:
                t_max = t
        return t_max

    def budgets(self, reverse=False):
        return sorted(self._targets, reverse=reverse)

    def budget(self, t):
        """
        Get the budget associated with a given rung.

        Parameters
        ----------
        t : int
            A nonnegative integer representing the rung.

        Returns
        -------
        float
            The budget associated with the given rung.
        """
        budgets = self.budgets()
        return budgets[t]

    def _rung_size_from_budget(self, b):
        # return sum(b in dct for dct in self._data.values())
        return len(self._targets[b])

    def rung_sizes(self):
        return [self._rung_size_from_budget(b) for b in self.budgets()]

    def rung_size(self, t):
        b = self.budget(t)
        return self._rung_size_from_budget(b)

    def size(self):
        return sum(self.rung_sizes())

    def load_feature_matrix(self):
        return np.vstack(self._data)

    def num_features(self):
        return len(self._data)

    def _targets_from_budget(self, b):
        return self._targets[b]

    def targets(self, t):
        b = self.budget(t)
        return self._targets_from_budget(b)

    def _threshold_from_budget(self, b):
        targets = self._targets_from_budget(b)
        tau = np.quantile(targets, q=self.gamma)
        return tau

    def threshold(self, t):
        b = self.budget(t)
        return self._threshold_from_budget(b)

    def thresholds(self):
        return [self._threshold_from_budget(b) for b in self.budgets()]

    def _binary_labels_from_budget(self, b):
        targets = self._targets_from_budget(b)
        tau = self._threshold_from_budget(b)
        return np.less_equal(targets, tau)

    def binary_labels(self, t):
        b = self.budget(t)
        return self._binary_labels_from_budget(b)

    def sequences_dict(self, pad_value=-1., binary=True, return_indices=False):
        """
        Create a dictionary of target sequences (lists of target labels of
        varying length), with the corresponding input (represented by a tuple
        of floats) as the key.
        Parameters
        ----------
        gamma : float, optional
            If not specified (default), returns sequences of continuous-valued
            targets. Otherwise, returns *binary* labels indicating whether the
            value is within the first `gamma`-quantile of all values observed
            at the same rung.
        Returns
        -------
        dict
            A dictionary of target sequences.
        """
        assert not binary or self.gamma is not None, \
            "Must instantiate with `gamma` specified for binary labels!"

        sequences = {}
        indices = {}
        for t, b in enumerate(self.budgets()):
            tau = self._threshold_from_budget(b)
            for k, dct in self._data.items():
                ys = sequences.setdefault(k, [])
                ind = indices.setdefault(k, [])
                # return np.less_equal(targets, tau)
                pred = (b in dct)
                if pred:
                    value = dct[b]
                    y = int(value <= tau) if binary else value
                else:
                    y = pad_value
                ind.append(pred)
                ys.append(y)

        if return_indices:
            return sequences, indices
        else:
            return sequences

    def sequences(self, pad_value=-1., binary=True):

        sequences, indices = self.sequences_dict(pad_value=pad_value,
                                                 binary=binary,
                                                 return_indices=True)

        input_sequences = []
        target_sequences = []
        for k, ys in sequences.items():

            T = len(ys)
            D = len(k)

            ind = indices[k]
            x = np.array(k)

            input_sequence = np.full(shape=(T, D), fill_value=pad_value,
                                     dtype="float64")
            input_sequence[ind] = x
            input_sequences.append(input_sequence)

            target_sequence = np.expand_dims(ys, axis=-1)
            target_sequences.append(target_sequence)

        inputs = np.stack(input_sequences, axis=0)
        targets = np.stack(target_sequences, axis=0)
        return inputs, targets

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        # Clever ways of doing this would involve data structs. like KD-trees
        # or locality sensitive hashing (LSH), but these are premature
        # optimizations at this point, especially since the `any` below does
        # lazy evaluation, i.e. is early stopped as soon as anything
        # returns `True`.
        # TODO(LT): We only need to look at the lowest rung.
        return any(np.allclose(np.array(k), x, rtol=rtol, atol=atol)
                   for k in self._data)
