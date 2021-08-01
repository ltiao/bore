import numpy as np


from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

Evaluations = namedtuple('Evaluations', ["targets", "inputs"])


class Record:

    def __init__(self, gamma=None):
        self.rungs = {}
        self.gamma = gamma

    @staticmethod
    def get_key(x):
        return tuple(x.tolist())

    def append(self, x, y, b):
        e = self.rungs.setdefault(b, Evaluations(targets=[], inputs=[]))
        e.inputs.append(x)
        e.targets.append(y)

    def test(self):

        inputs = []
        for b in self.rungs:
            inputs.extend(self.rungs[b].inputs)
        return np.vstack(inputs)

    def num_rungs(self):
        """
        Get the total number of rungs recorded.
        Returns
        -------
        int
            Total number of rungs recorded.
        """
        return len(self.rungs)

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
        for t, b in enumerate(sorted(self.rungs)):
            if self._rung_size_from_budget(b) >= min_size:
                t_max = t
        return t_max

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

    def _threshold_from_budget(self, b):
        tau = np.quantile(self.rungs[b].targets, q=self.gamma)
        return tau

    def threshold(self, t):
        b = self.budget(t)
        return self._threshold_from_budget(b)

    def thresholds(self):
        return [self._threshold_from_budget(b) for b in self.budgets()]

    def _binary_labels_from_budget(self, b):
        tau = self._threshold_from_budget(b)
        return np.less(self.rungs[b].targets, tau)

    def binary_labels(self, t):
        b = self.budget(t)
        return self._binary_labels_from_budget(b)

    def sequences_dict(self, binary=True):
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
        for b in sorted(self.rungs):

            if binary:
                targets = self._binary_labels_from_budget(b)
            else:
                targets = self.rungs[b].targets

            for x, y in zip(self.rungs[b].inputs, targets):
                key = self.get_key(x)
                ys = sequences.setdefault(key, [])
                ys.append(y)

                assert len(ys) <= self.num_rungs(), \
                    (f"Sequence length is {len(ys)} but we've only observed "
                     f"data for {self.num_rungs()} rung(s). "
                     "There must be duplicate input feature vectors!")

        return sequences

    def sequences(self, binary=True):
        """
        Create a pair of lists containing 2-D input and target sequences of
        shapes ``(t_n, d)`` and ``(t_n, 1)``, respectively, where ``t_n`` is
        the length of the `n`-th sequence.
        Parameters
        ----------
        gamma : float, optional
            If not specified (default), returns sequences of continuous-valued
            targets. Otherwise, returns *binary* labels indicating whether the
            value is within the first `gamma`-quantile of all values observed
            at the same rung.
        Returns
        -------
        input_sequences : list of array_like
            A list of 2-D input arrays with shapes ``(t_n, d)``.
        target_sequences : list of array_like
            A list of 2-D target arrays with shapes ``(t_n, 1)``.
        """
        sequences = self.sequences_dict(binary=binary)

        input_sequences = []
        target_sequences = []
        for x, ys in sequences.items():
            # input sequence array from shape (D,) to (t_n, D) by repeating
            xb = np.expand_dims(x, axis=0)  # broadcast `x` before repeating
            input_sequence = np.repeat(xb, repeats=len(ys), axis=0)
            input_sequences.append(input_sequence)
            # target sequence array from shape (t_n,) to (t_n, 1)
            target_sequence = np.expand_dims(ys, axis=-1)
            target_sequences.append(target_sequence)

        return input_sequences, target_sequences

    def sequences_padded(self, pad_value=1e+9, binary=True):
        """
        Create a pair of 3-D arrays of input and target sequences of shapes
        ``(N, t_max, d)`` and ``(N, t_max, 1)``, respectively, where ``N`` is
        the number of unique input feature vectors observed so far, and
        ``t_max = max(t_n for n in N)``.
        Parameters
        ----------
        gamma : float, optional
            If not specified (default), returns sequences of continuous-valued
            targets. Otherwise, returns *binary* labels indicating whether the
            value is within the first `gamma`-quantile of all values observed
            at the same rung.
        pad_value : float, optional
            The value used to pad undefined entries. It is important to specify
            a value that is unlikely to appear as an input feature, for example
            an implausibly small or large value (default 1e+9).
        Returns
        -------
        inputs : array_like
            A 3-D array of padded input sequences with shape ``(N, t_max, d)``.
        targets : array_like
            A 3-D array of padded target sequences with shape ``(N, t_max, 1)``.
        """
        input_sequences, target_sequences = self.sequences(binary=binary)
        target_dtype = "int32" if binary else "float64"
        return (pad_sequences(input_sequences, dtype="float64",
                              padding="post", value=pad_value),
                # TODO(LT): We don't strictly need to use `pad_sequences` for
                # the targets. The fact that we are using it might cause some
                # confusion...
                pad_sequences(target_sequences, dtype=target_dtype,
                              padding="post", value=pad_value))

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        # Clever ways of doing this would involve data structs. like KD-trees
        # or locality sensitive hashing (LSH), but these are premature
        # optimizations at this point, especially since the `any` below does
        # lazy evaluation, i.e. is early stopped as soon as anything
        # returns `True`.
        # TODO(LT): We only need to look at the lowest rung.
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol)
                   for b in self.rungs for x_prev in self.rungs[b].inputs)
