import torch

from torch import nn
from torch.distributions.uniform import Uniform
from scipy.optimize import minimize

from .nn_classifier import NN
from .blr_classifier import BayesianLogisticRegression
from .stein import SVN, scaled_hessian_RBF

from .utils.plotting import plot_iteration
from .utils.LBFGS import FullBatchLBFGS


class BORE(nn.Module):
    def __init__(self,
                 objective=None,
                 classifier=None,
                 constraints=None,
                 gamma=torch.tensor(0.25),
                 dim=1,
                 niter=20,
                 nepoch=200,
                 batch=False,
                 nbatch=5,
                 ninit=10,
                 normalise=False,
                 refine=False,
                 refine_niter=50,
                 gpu=False,
                 verbose=False):

        super(BORE, self).__init__()
        self.nepoch = nepoch
        self.niter = niter
        self.dim = dim
        self.x_star = torch.rand([1, self.dim]).requires_grad_(True)
        self.optimiser = FullBatchLBFGS(params=[self.x_star], lr=1., line_search='None')
        self.gamma = torch.tensor(gamma)
        self.refine = refine
        self.refine_niter = refine_niter  # number of function evaluations for refinement
        self.classifier = classifier
        self.batch = batch  # whether to do batch Stein optimization
        self.nbatch = nbatch  # number of Stein particles for batch optimization
        if constraints is not None:
            self.constraints = torch.tensor(constraints)
        else:
            self.constraints = None
        self.objective = objective
        self.X = torch.rand([ninit, self.dim])
        self.y = self.objective(self.X).view(-1)
        self.z = torch.zeros_like(self.y)
        self.verbose = verbose

    def step(self):
        # Assign labels according to gamma
        indices = self.quantiles_from_samples(self.y, self.gamma)
        self.z = torch.zeros_like(self.y)
        self.z[indices] = 1.

        # Trains the classifier on current data wrt BCE loss
        self.classifier.fit(self.X, self.z, nepoch=self.nepoch, batch_size=None)

        # Finds the maximum of the classifier and collects new candidates
        if self.batch:
            x_new = self.stein_batch_inference(niter=20)
        else:
            x_new = self.find_maximum_of_classifier(niter=20)
        y_new = self.objective(x_new)

        # Updates the dataset
        self.X = torch.cat((self.X, x_new), dim=0)
        self.y = torch.cat((self.y, y_new.view(-1)), dim=0)

        # Updates gamma
        if self.gamma > 0.05:
            self.gamma *= 0.9

    def optimise(self, plot=False):
        for i in range(self.niter):
            self.step()
            # Plot the iteration
            if plot:
                plot_iteration(self.X, self.y, self.z, i, self.classifier, self.objective)

        ymin, index = torch.min(self.y, 0)
        if self.verbose:
            print("Solution before refinement")
            print(f"x_t = {self.X[index, :].detach().numpy()}",
                  f"fmin = {ymin.detach().numpy()}")

        if self.refine:
            return self.refine_solution(self.X[index, :])

        return ymin, self.X[index, :]

    def quantiles_from_samples(self, values, quantile):
        _, indices = torch.sort(values.reshape(-1))
        q = torch.floor(quantile * len(values)).long()
        if q < 1:
            q = int(1)
        return indices[0:q]

    def find_maximum_of_classifier(self, niter=20):
        # self.x_star = self.x_star.detach().requires_grad_(True)
        self.x_star = torch.rand([1, self.dim])

        # Scale to the correct range
        if self.constraints is not None:
            self.x_star[0, :] = self.constraints[:, 0] + self.x_star[0, :] * \
                                (self.constraints[:, 1] - self.constraints[:, 0])

        self.x_star = self.x_star.requires_grad_(True)
        # self.optimiser = FullBatchLBFGS(params=[self.x_star], lr=1., line_search='None')
        self.optimiser = torch.optim.RMSprop(params=[self.x_star], lr=.01)

        if isinstance(self.classifier, NN) or \
                isinstance(self.classifier, BayesianLogisticRegression):
            for i in range(niter):
                """
                Gradient-based optimisation
                """
                self.optimiser.zero_grad()
                loss = - self.classifier(self.x_star)
                loss.backward()
                self.optimiser.step()
                if self.constraints is not None:
                    with torch.no_grad():
                        for j in range(self.dim):
                            self.x_star[0, j].clamp_(self.constraints[j, 0].item(),
                                                     self.constraints[j, 1].item())

        else:
            """
            Uses global optimisers in scipy
            """
            options = {'maxiter': 200}
            constraints = []

            if self.constraints is not None:
                for i in range(self.dim):
                    constraints.append({"fun": lambda x: - self.constraints[i, 0].numpy() + x[i], "type": "ineq"})
                    constraints.append({"fun": lambda x: self.constraints[i, 1].numpy() - x[i], "type": "ineq"})

            def _numpy_objective(x):
                x = torch.tensor(x, dtype=torch.get_default_dtype())
                loss = - self.classifier(x)
                return loss.detach().numpy()

            res = minimize(_numpy_objective, self.x_star.reshape([self.dim]).detach().numpy(),
                           method='COBYLA',
                           tol=1e-3,
                           options=options,
                           constraints=constraints)

            self.x_star = torch.tensor(res.x, dtype=torch.get_default_dtype()).reshape([1, self.dim])

        if self.verbose:
            print(f"BORE: x_t = {self.x_star.detach().numpy()}")
        return self.x_star.detach()

    def stein_batch_inference(self, niter=20):
        self.x_star = torch.rand([self.nbatch, self.dim])

        # Scale to the correct range
        if self.constraints is not None:
            self.x_star[0, :] = self.constraints[:, 0] + self.x_star[0, :] * \
                                (self.constraints[:, 1] - self.constraints[:, 0])

        self.x_star = self.x_star.requires_grad_(True)
        stein_optimiser = FullBatchLBFGS(params=[self.x_star], lr=.05, line_search='None')
        K = scaled_hessian_RBF(sigma=None)

        def _log_constraints(x):
            if self.constraints is not None:
                dist = Uniform(self.constraints[:, 0], self.constraints[:, 1])
                return torch.sum(dist.log_prob(x), dim=1)
            else:
                return 0.

        def _log_classifier(x):
            return torch.log(self.classifier(x)) + _log_constraints(x)

        svn = SVN(x=self.x_star,
                  kernel=K,
                  optimizer=stein_optimiser,
                  log_p=_log_classifier,
                  constraints=self.constraints)

        if isinstance(self.classifier, NN) or \
                isinstance(self.classifier, BayesianLogisticRegression):
            for i in range(niter):
                """
                Stein-based optimisation
                """
                svn.step(self.x_star, alpha=1.)

                if self.constraints is not None:
                    with torch.no_grad():
                        for j in range(self.dim):
                            self.x_star[0, j].clamp_(self.constraints[j, 0].item(),
                                                     self.constraints[j, 1].item())
        else:
            ValueError("Not possible to run Stein optimization with this classifier.")
        if self.verbose:
            print(f"BORE: x_t = {self.x_star.detach().numpy()}")
        return self.x_star.detach()

    def refine_solution(self, x):
        options = {'maxiter': self.refine_niter, 'catol': 1.e-3}
        constraints = []

        if self.constraints is not None:
            for i in range(self.dim):
                constraints.append({"fun": lambda x: - self.constraints[i, 0].numpy() + x[i], "type": "ineq"})
                constraints.append({"fun": lambda x: self.constraints[i, 1].numpy() - x[i], "type": "ineq"})

        def _numpy_objective(x):
            x = torch.tensor(x, dtype=torch.get_default_dtype()).reshape([1, self.dim])
            loss = self.objective(x)  # - \
            self.X = torch.cat((self.X, x), dim=0)
            self.y = torch.cat((self.y, loss.detach().view(1)), dim=0)
            return loss.detach().numpy()

        res = minimize(_numpy_objective, x.reshape([self.dim]).detach().numpy(),
                       method='COBYLA',
                       tol=1e-3,
                       options=options,
                       constraints=constraints)

        xmin = torch.tensor(res.x, dtype=torch.float32).reshape([1, self.dim])
        ymin = self.objective(xmin.reshape([1, self.dim]))
        if self.verbose:
            print("Solution after refinement")
            print(f"x_t = {xmin.detach().numpy()}",
                  f"fmin = {ymin.detach().numpy()}")

        return ymin, xmin
