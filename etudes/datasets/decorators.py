import numpy as np


def binarize(positive_label=3, negative_label=5):
    """
    MNIST binary classification.

    Examples
    --------

    .. plot::
        :context: close-figs

        import tensorflow as tf

        from etudes.datasets import binarize
        from etudes.plotting import plot_image_grid

        @binarize(positive_label=2, negative_label=7)
        def binary_mnist_load_data():
            return tf.keras.datasets.mnist.load_data()

        (X_train, Y_train), (X_test, Y_test) = binary_mnist_load_data()

        num_train, img_rows, img_cols = X_train.shape
        num_test, img_rows, img_cols = X_test.shape

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        plot_image_grid(ax1, X_train[Y_train == 0],
                        shape=(img_rows, img_cols), nrows=10, cmap="cividis")

        plot_image_grid(ax2, X_train[Y_train == 1],
                        shape=(img_rows, img_cols), nrows=10, cmap="cividis")

        plt.show()
    """

    # TODO: come up with remote descriptive name
    def d(X, y, label, new_label=1):

        X_val = X[y == label]
        y_val = np.full(len(X_val), new_label)

        return X_val, y_val

    def binarize_decorator(load_data_fn):

        def new_load_data_fn():

            (X_train, Y_train), (X_test, Y_test) = load_data_fn()

            X_train_pos, Y_train_pos = d(X_train, Y_train,
                                         label=positive_label, new_label=1)
            X_train_neg, Y_train_neg = d(X_train, Y_train,
                                         label=negative_label, new_label=0)

            X_train_new = np.vstack([X_train_pos, X_train_neg])
            Y_train_new = np.hstack([Y_train_pos, Y_train_neg])

            X_test_pos, Y_test_pos = d(X_test, Y_test,
                                       label=positive_label, new_label=1)
            X_test_neg, Y_test_neg = d(X_test, Y_test,
                                       label=negative_label, new_label=0)

            X_test_new = np.vstack([X_test_pos, X_test_neg])
            Y_test_new = np.hstack([Y_test_pos, Y_test_neg])

            return (X_train_new, Y_train_new), (X_test_new, Y_test_new)

        return new_load_data_fn

    return binarize_decorator
