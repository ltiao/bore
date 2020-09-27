import ConfigSpace as CS
from hyperopt import hp


def config_space_to_search_space(config_space, q=1):
    """
    Converts HpBandSter's ConfigurationSpace to HyperOpt's search space
    dictionary format.
    """
    search_space = {}
    for h in config_space.get_hyperparameters():
        if isinstance(h, CS.OrdinalHyperparameter):
            search_space[h.name] = hp.quniform(h.name, 0, len(h.sequence)-1, q)
        elif isinstance(h, CS.CategoricalHyperparameter):
            search_space[h.name] = hp.choice(h.name, h.choices)
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            search_space[h.name] = hp.quniform(h.name, h.lower, h.upper, q)
        elif isinstance(h, CS.UniformFloatHyperparameter):
            search_space[h.name] = hp.uniform(h.name, h.lower, h.upper)

    return search_space


def kwargs_to_config(kwargs, config_space):

    config = {}
    for h in config_space.get_hyperparameters():
        if isinstance(h, CS.OrdinalHyperparameter):
            value = h.sequence[int(kwargs[h.name])]
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            value = int(kwargs[h.name])
        else:
            value = kwargs[h.name]
        config[h.name] = value

    return config
