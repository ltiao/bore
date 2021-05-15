"""
# Bayesian Logistic Regression in pytorch using the Laplace approximation
# Implementation based on:
# https://github.com/RansML/Bayesian_Hilbert_Maps/blob/master/BHM/pytorch/bhmtorch_cuda.py
# and Bishop's book "Pattern Recognition and Machine Learning"
"""
import numpy as np
import torch

from torch import nn
from .utils.random_fourier_features import RFF

dtype = torch.float32
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU


class BayesianLogisticRegression(nn.Module):
    def __init__(self,
                 dim=1,
                 gamma=10.,
                 nfeat=100,
                 cosOnly=False,
                 quasiRandom=True,
                 kernel="RBF",
                 gpu=False,
                 nIter=0):
        """
        :param gamma: RBF bandwidth
        """
        super(BayesianLogisticRegression, self).__init__()
        self.gamma = gamma
        self.nIter = nIter  # number of training iterations
        self.rff = RFF(nfeat, dim, 1./self.gamma, cosOnly, quasiRandom, kernel, gpu)

    def __rbf_kernel(self, X1, X2, gamma):

        K = torch.norm(X1[:, None] - X2, dim=-1, p=2).pow(2)
        K = torch.exp(-gamma * K)

        return K

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = self.__rbf_kernel(X, self.grid, gamma=self.gamma)
        rbf_features = torch.cat((torch.ones(X.shape[0], 1).cuda(), rbf_features), dim=1)

        return rbf_features

    def __calc_features(self, X):
        return self.rff.toFeatures(X)

    def __calc_posterior(self, X, y, epsilon, mu0, sig0):
        """
        :param X: input features
        :param y: labels
        :param epsilon: per dimension local linear parameter
        :param mu0: mean
        :param sig0: variance
        :return: new_mean, new_varaiance
        """
        logit_inv = torch.sigmoid(epsilon)
        lambda_ = 0.5 / epsilon * (logit_inv - 0.5)  # Eq 4 appendix of BHM paper
        sig = 1. / (1. / sig0 + 2. * torch.sum((X.t() ** 2) * lambda_, dim=1))  # Eq 12 appendix of BHM paper
        mu = sig * (mu0 / sig0 + torch.mm(X.t(), y.reshape(-1, 1) - 0.5).squeeze())  # Eq 11 appendix of BHM paper
        return mu, sig

    def fit(self, X, y, nepoch=10, batch_size=None):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__calc_features(X)
        N, D = X.shape[0], X.shape[1]

        self.epsilon = torch.ones(N, dtype=torch.float32)
        if not hasattr(self, 'mu'):
            self.mu = torch.zeros(D, dtype=torch.float32)
            self.sig = 1e-2 * torch.ones(D, dtype=torch.float32)  # Initial prior discussed in appendix of BHM paper

        for i in range(nepoch):
            #print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = torch.sqrt(
                torch.sum((X ** 2) * self.sig, dim=1) +
                (X.mm(self.mu.reshape(-1, 1)) ** 2).squeeze())  # Eq 9 appendix BHM paper

    def predict(self, Xq):
        """
        :param Xq: raw inquery points
        :return: mean classification (Laplace approximation)
        """
        Xq = self.__calc_features(Xq)  # Calculate features
        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()  # Eq. 4.149 Bishop's book
        sig2_inv_a = torch.sum((Xq ** 2) * self.sig, dim=1)  # Eq. 4.150 Bishop's book
        k = 1.0 / torch.sqrt(1. + np.pi * sig2_inv_a / 8.)  # Eq. 4.154 Bishop's book

        return torch.sigmoid(k * mu_a)  # Eq. 4.155 Bishop's book

    def predict_ucb(self, Xq):
        """
                :param Xq: raw inquery points
                :return: mean classification (Laplace approximation)
                """
        ucb_param = 3.
        Xq = self.__calc_features(Xq)  # Calculate features
        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()  # Eq. 4.149 Bishop's book
        sig2_inv_a = torch.sum((Xq ** 2) * self.sig, dim=1)  # Eq. 4.150 Bishop's book
        k = 1.0 / torch.sqrt(1. + np.pi * sig2_inv_a / 8.)  # Eq. 4.154 Bishop's book

        pred = torch.sigmoid(k * mu_a)  # Eq. 4.155 Bishop's book
        return pred + ucb_param * torch.sqrt(sig2_inv_a)

    def forward(self, X, ucb=True):
        if not ucb:
            return self.predict(X)
        else:
            return self.predict_ucb(X)

    def predictSampling(self, Xq, nSamples=50):
        """
        :param Xq: raw inquery points
        :param nSamples: number of samples to take the average over
        :return: sample mean and standard deviation of occupancy
        """
        Xq = self.__calc_features(Xq)

        qw = torch.distributions.MultivariateNormal(self.mu, torch.diag(self.sig))
        w = qw.sample((nSamples,)).t()

        mu_a = Xq.mm(w).squeeze()
        probs = torch.sigmoid(mu_a)

        mean = torch.std(probs, dim=1).squeeze()
        std = torch.std(probs, dim=1).squeeze()

        return mean, std
