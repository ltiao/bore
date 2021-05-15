# Wrapper for scikit-learn regression
import numpy as np
import sklearn
import torch

from torch import nn
from sklearn.ensemble import BaseEnsemble, RandomForestRegressor
from sklearn.linear_model import ARDRegression

import sklearn.neighbors


class SKRegression(nn.Module):
    def __init__(self, regressor='RandomForestRegressor', gamma=1., dim=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        if regressor == 'RandomForestRegressor':
            self.regressor = getattr(sklearn.ensemble, regressor, kwargs)()
        elif regressor == 'BayesianLinearRegression':
            self.regressor = getattr(sklearn.linear_model, "ARDRegression", kwargs)()

    def forward(self, x):
        if isinstance(self.regressor, RandomForestRegressor):
            pred = np.stack([e.predict(x.reshape([-1, self.dim]).detach().numpy()) for e in
                             self.regressor.estimators_])

            return torch.as_tensor(pred.mean(axis=0) - self.gamma * pred.std(axis=0))

        predictions = self.regressor.predict(x.reshape([-1, self.dim]).detach().numpy())
        return torch.tensor(predictions[:, 1])

    def fit(self, x_data, y_data, nepoch=1000, batch_size=50, verbose=True):
        self.regressor.fit(X=x_data.detach().numpy(), y=y_data.detach().numpy().ravel())
