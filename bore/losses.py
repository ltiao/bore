from tensorflow.keras.losses import binary_crossentropy


def binary_crossentropy_from_logits(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred, from_logits=True)
