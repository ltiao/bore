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
