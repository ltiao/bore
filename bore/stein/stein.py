import torch
import numpy as np


class SVN:
    """
    Stein Variational Newton Method
    """

    def __init__(self, x, kernel, optimizer, log_p, constraints):
        self.kernel = kernel
        self.optimizer = optimizer
        self.log_p = log_p
        self.constraints = constraints
        self.H_diag = None
        self.nParticles = None
        self.dim = None
        self.stepsize = 1.
        self.maxmaxshiftold = torch.tensor(float('inf'))
        self.maxmaxshiftold_np = np.inf
        self.maxshift = np.zeros(self.nParticles)
        self.H = torch.zeros(x.size(0), x.size(1), x.size(1))
        self.H_block_loss = torch.zeros(x.size(0), x.size(1), x.size(1))
        self.H_loss = None

    def phi(self, x):
        x = x.detach().requires_grad_(True)
        loss = -self.log_p(x).sum()

        if self.dim is None:
            self.dim = x.size(1)
        if self.nParticles is None:
            self.nParticles = x.size(0)

        minus_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        M = torch.mean(self.GaussNewtonHessian(minus_grad), dim=0)
        k_xx, grad_k = self.kernel(x.detach(), x.detach(), M)
        phi = (k_xx.detach().matmul(- minus_grad) + grad_k) / self.nParticles
        return -phi, loss

    def fix_constraints(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[1])
        for i in range(x.shape[1]):
            # r = self.constraints[i, 1] - self.constraints[i, 0]
            x[x[:, i] < self.constraints[i, 0], i] = self.constraints[i, 0] #+ 0.5 * r
            x[x[:, i] > self.constraints[i, 1], i] = self.constraints[i, 1] #- 0.5 * r
        return x.reshape(orig_shape)

    def step(self, x, alpha):
        """
        For LBFGS
        """
        def closure():
            self.optimizer.zero_grad()
            x.grad, loss = self.phi(x)
            return loss

        loss = closure()
        options = {'closure': closure, 'current_loss': loss}
        self.optimizer.step(options)
        if self.constraints is not None:
            x = self.fix_constraints(x)

    def GaussNewtonHessian(self, J):
        return 2 * J.reshape([self.nParticles, self.dim, 1]) * \
               J.reshape([self.nParticles, 1, self.dim])

    def resetParticles(self, x):
        x = torch.randn(self.nParticles, self.dim)
        return x


class scaled_hessian_RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(scaled_hessian_RBF, self).__init__()
        self.sigma = sigma

    def __call__(self, X, Y, metric=None):
        if torch.is_tensor(metric):
            self.sigma = metric
        else:
            self.sigma = torch.eye(X.shape[1])

        def compute_script(X, Y, sigma):
            K_XY = torch.zeros(X.size(0), Y.size(0))
            dK_XY = torch.zeros(X.shape)
            for i in range(X.shape[0]):
                sign_diff = Y[i, :] - X
                Msd = torch.matmul(sign_diff, sigma)
                K_XY[i, :] = torch.exp( - 0.5 * torch.sum(sign_diff * Msd, 1))
                dK_XY[i] = K_XY[i, :].matmul(Msd) * 2
            return K_XY, dK_XY

        K_XY, dK_XY = compute_script(X, Y, self.sigma)
        return K_XY, dK_XY