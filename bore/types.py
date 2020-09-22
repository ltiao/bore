import numpy as np
import ConfigSpace as CS

from scipy.optimize import Bounds


class DenseConfigurationSpace(CS.ConfigurationSpace):

    def __init__(self, other, *args, **kwargs):

        super(DenseConfigurationSpace, self).__init__(*args, **kwargs)
        # deep-copy only the hyperparameters. conditions, clauses, seed,
        # and other metadata ignored
        self.add_hyperparameters(other.get_hyperparameters())

        nums, cats, size_sparse, size_dense = self.get_mappings()

        if nums:
            self.num_src, self.num_trg = map(np.uintp, zip(*nums))

        if cats:
            self.cat_src, self.cat_trg, self.cat_sizes = \
                map(np.uintp, zip(*cats))

        self.nums = nums
        self.cats = cats
        self.size_sparse = size_sparse
        self.size_dense = size_dense

    def sample_configuration(self, size=1):
        configs = super(DenseConfigurationSpace, self).sample_configuration(size=size)
        configs = [configs] if size == 1 else configs
        for config in configs:
            yield DenseConfiguration(self, values=config)

    def get_bounds(self):

        # lowers = []
        # uppers = []
        # for hp in self.get_hyperparameters():
        #     if isinstance(hp, CS.CategoricalHyperparameter):
        #         cat_size = hp.num_choices
        #         for i in range(cat_size):
        #             lowers.append(0.)
        #             uppers.append(1.)
        #     elif isinstance(hp, CS.UniformFloatHyperparameter):
        #         lowers.append(hp._inverse_transform(hp.lower))
        #         uppers.append(hp._inverse_transform(hp.upper))
        #     elif isinstance(hp, CS.UniformIntegerHyperparameter):
        #         lowers.append(hp._inverse_transform(hp.lower - 1))
        #         uppers.append(hp._inverse_transform(hp.upper + 1))
        #     else:
        #         raise NotImplementedError(
        #             "Only hyperparameters of types "
        #             "`CategoricalHyperparameter`, "
        #             "`UniformIntegerHyperparameter`, "
        #             "`UniformFloatHyperparameter` are supported!")
        # assert len(lowers) == self.size_dense
        # assert len(uppers) == self.size_dense

        # All of the above commented code is equivalent to the following two
        # lines...
        lowers = np.zeros(self.size_dense)
        uppers = np.ones(self.size_dense)

        # return list(zip(lowers, uppers))
        return Bounds(lowers, uppers)

    def get_mappings(self):

        nums = []
        cats = []

        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                cats.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
            elif isinstance(hp, (CS.UniformIntegerHyperparameter,
                                 CS.UniformFloatHyperparameter)):
                nums.append((src_ind, trg_ind))
                trg_ind += 1
            else:
                raise NotImplementedError(
                    "Only hyperparameters of types "
                    "`CategoricalHyperparameter`, "
                    "`UniformIntegerHyperparameter`, "
                    "`UniformFloatHyperparameter` are supported!")

        size_sparse = src_ind + 1
        size_dense = trg_ind

        return nums, cats, size_sparse, size_dense


class DenseConfiguration(CS.Configuration):

    def __init__(self, configuration_space, *args, **kwargs):

        assert isinstance(configuration_space, DenseConfigurationSpace)
        super(DenseConfiguration, self).__init__(configuration_space,
                                                 *args, **kwargs)

    @classmethod
    def from_array(cls, configuration_space, array_dense):

        assert isinstance(configuration_space, DenseConfigurationSpace)
        cs = configuration_space
        # TODO(LT): specify `dtype` flexibly
        # initialize output array
        array_sparse = np.empty(cs.size_sparse, dtype="float64")

        # process numerical hyperparameters
        if cs.nums:
            array_sparse[cs.num_src] = array_dense[cs.num_trg]

        # process categorical hyperparameters
        for src_ind, trg_ind, size in cs.cats:
            ind_max = np.argmax(array_dense[trg_ind:trg_ind + size])
            array_sparse[src_ind] = ind_max

        return cls(configuration_space=configuration_space, vector=array_sparse)

    def to_array(self):

        cs = self.configuration_space
        array_sparse = super(DenseConfiguration, self).get_array()

        # TODO(LT): specify `dtype` flexibly
        # initialize output array
        array_dense = np.zeros(cs.size_dense, dtype="float64")

        # process numerical hyperparameters
        if cs.nums:
            array_dense[cs.num_trg] = array_sparse[cs.num_src]

        # process categorical hyperparameters
        if cs.cats:
            cat_trg_offset = np.uintp(array_sparse[cs.cat_src])
            array_dense[cs.cat_trg + cat_trg_offset] = 1

        return array_dense
