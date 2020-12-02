from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, RNN, Dense, Input, LSTMCell,
                                     Masking, RepeatVector, TimeDistributed)


class DenseSequential(Sequential):

    def __init__(self, input_dim, output_dim, num_layers, num_units,
                 layer_kws={}, final_layer_kws={}):

        super(DenseSequential, self).__init__()

        for i in range(num_layers):
            if not i:
                self.add(Dense(num_units, input_dim=input_dim, **layer_kws))
            self.add(Dense(num_units, **layer_kws))

        self.add(Dense(output_dim, **final_layer_kws))
