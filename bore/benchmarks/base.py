from abc import ABC, abstractmethod
from collections import namedtuple

from ..utils import config_space_to_search_space, config_space_to_domain

Evaluation = namedtuple('Evaluation', ['value', 'duration'])


class BenchmarkBase(ABC):

    @abstractmethod
    def get_config_space(self):
        pass

    @abstractmethod
    def get_minimum(self):
        pass


class Benchmark(BenchmarkBase):

    def get_search_space(self):
        cs = self.get_config_space()
        return config_space_to_search_space(cs)

    def get_domain(self):
        cs = self.get_config_space()
        return config_space_to_domain(cs)
