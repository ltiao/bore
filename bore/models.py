from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DenseSequential(Sequential):

    def __init__(self, output_dim, num_layers, num_units, layer_kws={},
                 final_layer_kws={}):

        super(DenseSequential, self).__init__()

        for l in range(num_layers):
            self.add(Dense(num_units, **layer_kws))

        self.add(Dense(output_dim, **final_layer_kws))
