=================
Matérn 5/2 Kernel
=================

.. plot::
   :context: close-figs
   :include-source:

    import numpy as np

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    import tensorflow_probability as tfp

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from etudes.gaussian_process import gp_sample_custom, dataframe_from_gp_samples

    # shortcuts
    tfd = tfp.distributions
    kernels = tfp.math.psd_kernels

    # constants
    n_features = 1 # dimensionality
    n_index_points = 256 # nbr of index points
    n_samples = 5 # nbr of GP prior samples 
    jitter = 1e-15
    kernel_cls = kernels.MaternFiveHalves

    seed = 42 # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

    # index points
    X_q = np.linspace(-1.0, 1.0, n_index_points).reshape(-1, n_features)

    # kernel specification
    amplitude, length_scale_inv = np.ogrid[0.05:0.16:0.05, 10.0:0.5:-1.5]
    length_scale = 1.0 / length_scale_inv
    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)

    # instantiate Gaussian Process
    gp = tfd.GaussianProcess(kernel=kernel, index_points=X_q, jitter=jitter)
    gp_samples = gp_sample_custom(gp, n_samples, seed=seed)

    with tf.Session() as sess:
        gp_samples_arr = sess.run(gp_samples)

    data = dataframe_from_gp_samples(gp_samples_arr, X_q, amplitude, 
                                     length_scale, n_samples)

.. plot::
   :context: close-figs
   :include-source:

    g = sns.relplot(x="index_point", y="function_value", hue="sample",
                    row="amplitude", col="length_scale", height=5.0, aspect=1.0,
                    kind="line", data=data, alpha=0.7, linewidth=3.0)
    g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
                 col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
    g.set_axis_labels(r"$x$", r"$f^{(i)}(x)$")

.. plot::
   :context: close-figs
   :include-source:

    g = sns.relplot(x="index_point", y="function_value", hue="length_scale",
                    row="amplitude", col="sample", height=5.0, aspect=1.0,
                    kind="line", data=data, alpha=0.7, linewidth=3.0)
    g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
                 col_template=r"sample {col_name}")
    g.set_axis_labels(r"$x$", r"$f^{(i)}(x)$")