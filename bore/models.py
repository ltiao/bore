import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Masking, RepeatVector,
                                     TimeDistributed, RNN, LSTMCell)

from .mixins import MaximizableMixin, BatchMaximizableMixin


class DenseSequential(Sequential):

    def __init__(self, input_dim, output_dim, num_layers, num_units,
                 layer_kws={}, final_layer_kws={}):

        super(DenseSequential, self).__init__()

        for i in range(num_layers):
            if not i:
                self.add(Dense(num_units, input_dim=input_dim, **layer_kws))
            self.add(Dense(num_units, **layer_kws))

        self.add(Dense(output_dim, **final_layer_kws))


class MaximizableModel(MaximizableMixin, Model):
    pass


class MaximizableSequential(MaximizableMixin, Sequential):
    pass


class MaximizableDenseSequential(MaximizableMixin, DenseSequential):
    pass


class BatchMaximizableModel(BatchMaximizableMixin, Model):
    pass


class BatchMaximizableSequential(BatchMaximizableMixin, Sequential):
    pass


class BatchMaximizableDenseSequential(BatchMaximizableMixin, DenseSequential):
    pass


class StackedRecurrentFactory:

    def __init__(self, input_dim, output_dim, num_layers=2, num_units=32,
                 layer_kws={}, final_layer_kws={}):

        # TODO(LT): Add support for any type of recurrent cells, e.g. GRUs.
        # TODO(LT): This implementation cannot take advantage of CuDNN,
        # since it operates at the layer level, not cell level.

        self.input_dim = input_dim

        assert "return_sequences" not in layer_kws
        assert "activation" not in final_layer_kws

        # Initialize stack of recurrent cells
        self.cells = []
        for i in range(num_layers):
            self.cells.append(LSTMCell(num_units, **layer_kws))
        # Initialize fully-connected final layer
        self.final_layer = Dense(output_dim, **final_layer_kws)

    def build_many_to_many(self, mask_value=1e+9):
        # At training time, we use a many-to-many network architecture.
        # Since the target sequences have varying lengths, we require an
        # input masking layer.
        # For numerical stability, we don't explicitly use an sigmoid
        # output activation. Instead, we rely on the the `from_logits=True`
        # option in the loss constructor.
        input_shape = (None, self.input_dim)

        network = Sequential()
        network.add(Masking(mask_value=mask_value, input_shape=input_shape))
        for cell in self.cells:
            network.add(RNN(cell, return_sequences=True))
        network.add(TimeDistributed(self.final_layer))
        return network

    def build_one_to_one(self, num_steps, transform=tf.identity):
        # At test time, we only care about the output at some particular step,
        # hence we use a one-to-one network, with a RepeatVector input layer,
        # and do not return sequences in the final recurrent layer.
        # When optimizing this network wrt to inputs, the final activation
        # can have a large effect on gradient magnitudes and, therefore,
        # convergence.
        input_shape = (self.input_dim,)
        num_layers = len(self.cells)

        network = MaximizableSequential(transform=transform)
        network.add(RepeatVector(num_steps, input_shape=input_shape))
        for i, cell in enumerate(self.cells):
            # equivalent to True if not final layer else False
            return_sequences = (i < num_layers - 1)
            network.add(RNN(cell, return_sequences=return_sequences))
        network.add(self.final_layer)
        # network.add(Activation(activation))

        return network
