from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from .mixins import MinimizableMixin


class MinimizableModel(MinimizableMixin, Model):
    pass


class MinimizableSequential(MinimizableMixin, Sequential):
    pass


class DenseMinimizableSequential(MinimizableMixin, Sequential):

    def __init__(self, input_dim, output_dim, num_layers, num_units,
                 layer_kws={}, final_layer_kws={}):

        super(DenseMinimizableSequential, self).__init__()

        for i in range(num_layers):
            if not i:
                self.add(Dense(num_units, input_dim=input_dim, **layer_kws))
            self.add(Dense(num_units, **layer_kws))

        self.add(Dense(output_dim, **final_layer_kws))
