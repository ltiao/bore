# Wrapper for scikit-learn classifiers
import numpy as np
import torch
import sklearn
import sklearn.neighbors

from torch import nn
from sklearn.ensemble import BaseEnsemble, RandomForestClassifier, GradientBoostingClassifier


class SKClass(nn.Module):
    def __init__(self, classifier='GradientBoostingClassifier', dim=1, ucb_param=3., **kwargs):
        super().__init__()
        self.dim = dim
        if classifier == 'GradientBoostingClassifier':
            self.classifier = getattr(sklearn.ensemble, classifier)()
        elif classifier == 'RandomForestClassifier' or \
                classifier == 'ExtraTreesClassifier':
            self.classifier = getattr(sklearn.ensemble, classifier, kwargs)()
        else:
            self.classifier = getattr(sklearn.neighbors, classifier, kwargs)()
        self.ucb_param = ucb_param

    def forward(self, x):
        if isinstance(self.classifier, RandomForestClassifier):
            pred = np.stack([e.predict_proba(x.reshape([-1, self.dim]).detach().numpy())[:, 1] for e in
                             self.classifier.estimators_])

            return torch.as_tensor(pred.mean(axis=0) + self.ucb_param * pred.std(axis=0))

        predictions = self.classifier.predict_proba(x.reshape([-1, self.dim]).detach().numpy())
        return torch.tensor(predictions[:, 1])

    def fit(self, x_data, y_data, nepoch=1000, batch_size=50, verbose=True):
        self.classifier.fit(X=x_data.detach().numpy(), y=y_data.detach().numpy().ravel())
