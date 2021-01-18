import numpy as np
import ConfigSpace as CS

from ..base import Benchmark, Evaluation
from .cars import F110
from .tracks import UCB
from .line import randomTrajectory, calcMinimumTime


class RacingLine(Benchmark):

    # fixed node at the end DO NOT CHANGE
    LASTIDX = 0
    # define indices for the nodes
    NODES = [10, 32, 44, 67, 83, 100, 113, 127, 144, 160, 175, 191]
    SCALE = 0.95
    N_WAYPOINTS = 100

    def __init__(self):
        # TODO(LT): Hard-coded for now
        self.track = UCB()
        self.car = F110()

        self.dimensions = len(self.NODES)
        self.theta = self.track.theta_track[self.NODES]

    def __call__(self, kwargs, budget=None):
        nodes = np.hstack([kwargs.get(f"node{d}") for d in range(self.dimensions)])
        lap_time = self.func(nodes)
        return Evaluation(value=lap_time, duration=None)

    def func(self, nodes):
        rand_traj = randomTrajectory(track=self.track, n_waypoints=self.dimensions)
        # nodes = rand_traj.sample_nodes(scale=self.scale)

        wx, wy = rand_traj.calculate_xy(width=nodes,
                                        last_index=self.NODES[self.LASTIDX],
                                        theta=self.theta)
        x, y = rand_traj.fit_cubic_splines(wx=wx, wy=wy, n_samples=self.N_WAYPOINTS)
        return calcMinimumTime(x, y, **self.car)

    def get_config_space(self):

        track_width = self.track.track_width * self.SCALE
        upper = 0.5 * track_width
        lower = upper - track_width

        cs = CS.ConfigurationSpace()
        for d in range(self.dimensions):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(f"node{d}", lower=lower, upper=upper))
        return cs

    def get_minimum(self):
        raise NotImplementedError
