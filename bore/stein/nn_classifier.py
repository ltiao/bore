import numpy as np
import torch

from torch import nn


class NN(nn.Module):
    """ Implements a simple neural network. """

    def __init__(self, hidden_layers=None, dim=1, normalise=False, gpu=False):
        super(NN, self).__init__()
        self.normalise = normalise
        self.network = []
        self.predictions = []
        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]
        self.hidden_layers = hidden_layers
        self.n_outputs = int(1)
        self.loss = nn.BCELoss()
        self.dim = dim
        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

        self.X = torch.tensor([]).type(self.dtype).to(self.device)
        self.y = torch.tensor([]).type(self.dtype).to(self.device)

        # Builds the neural network
        last_layer_size = None
        for ix, layer_size in enumerate(self.hidden_layers):

            if ix == 0:
                self.network.append(nn.Linear(self.dim, layer_size).
                                    to(self.device))
            else:
                self.network.append(nn.Linear(last_layer_size, layer_size).
                                    to(self.device))
            self.network.append(nn.ReLU().to(self.device))
            last_layer_size = layer_size

        self.mu = nn.Linear(last_layer_size, self.n_outputs).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    def forward(self, x):
        if self.normalise:
            x = (x - self.X_mean) / self.X_std

        z_h = None
        for ix, layer in enumerate(self.network):
            if ix == 0:
                z_h = layer(x)
            else:
                z_h = layer(z_h)

        predictions = self.mu(z_h).reshape(-1, self.n_outputs)
        predictions = self.sigmoid(predictions)

        return predictions

    def fit(self, x_data, y_data, nepoch=1000, batch_size=50, verbose=True):

        optimizer = torch.optim.RMSprop(self.parameters(), lr=.1, weight_decay=0.)
        # optimizer = torch.optim.Adam(self.parameters(), lr=.1, weight_decay=0.)

        x_variable = x_data
        y_variable = y_data

        batch_size = len(x_data) if batch_size is None else batch_size
        print("Training neural network on {} datapoints".format(len(x_data)))

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(x_data), batch_size)
                yield x_variable[indexes], y_variable[indexes]

        batch_gen_iter = batch_generator()

        for epoch in range(nepoch):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()
            predictions = self(x_batch)
            output = self.loss(predictions.reshape(-1), y_batch.reshape(-1))
            output.backward()
            optimizer.step()

            if epoch == 0:
                pass
                # print("Initial Loss is: {}".format(output))

            elif epoch % 100 == 0 and verbose:
                if epoch != 0:
                    pass
                    # print(" Iteration:", epoch, "Loss", output)
