from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from .mixins import MaximizableMixin


class MaximizableModel(MaximizableMixin, Model):
    pass


class MaximizableSequential(MaximizableMixin, Sequential):
    pass


class DenseMaximizableSequential(MaximizableMixin, Sequential):

    def __init__(self, transform, input_dim, output_dim, num_layers, num_units,
                 layer_kws={}, final_layer_kws={}):

        super(DenseMaximizableSequential, self).__init__(transform=transform)

        for i in range(num_layers):
            if not i:
                self.add(Dense(num_units, input_dim=input_dim, **layer_kws))
            self.add(Dense(num_units, **layer_kws))

        self.add(Dense(output_dim, **final_layer_kws))
