import numpy as np
import ConfigSpace as CS

from ..base import Benchmark, Evaluation
from .cars import F110, ORCA
from .tracks import UCB, ETHZ, ETHZMobil
from .line import randomTrajectory, calcMinimumTime


class RacingLine(Benchmark):

    def __init__(self, track, car, nodes, scale, lastidx, num_waypoints=100):
        self.num_waypoints = num_waypoints

        self.track = track
        self.car = car

        self.nodes = nodes
        self.scale = scale
        self.lastidx = lastidx

        self.dimensions = len(self.nodes)
        self.theta = self.track.theta_track[self.nodes]

    def __call__(self, kwargs, budget=None):
        nodes = np.hstack([kwargs.get(f"node{d}") for d in range(self.dimensions)])
        lap_time = self.func(nodes)
        return Evaluation(value=lap_time, duration=None)

    def func(self, nodes):
        rand_traj = randomTrajectory(track=self.track, n_waypoints=self.dimensions)
        wx, wy = rand_traj.calculate_xy(width=nodes,
                                        last_index=self.nodes[self.lastidx],
                                        theta=self.theta)
        x, y = rand_traj.fit_cubic_splines(wx=wx, wy=wy, n_samples=self.num_waypoints)
        return calcMinimumTime(x, y, **self.car)

    def get_config_space(self):

        track_width = self.track.track_width * self.scale
        upper = 0.5 * track_width
        lower = upper - track_width

        cs = CS.ConfigurationSpace()
        for d in range(self.dimensions):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(f"node{d}", lower=lower, upper=upper))
        return cs

    def get_minimum(self):
        raise NotImplementedError


class UCBF110RacingLine(RacingLine):

    def __init__(self):

        super(UCBF110RacingLine, self).__init__(
            track=UCB(),
            car=F110(),
            nodes=[10, 32, 44, 67, 83, 100, 113, 127, 144, 160, 175, 191],
            scale=0.95,
            lastidx=0,
            num_waypoints=100)


class ETHZORCARacingLine(RacingLine):

    def __init__(self):

        super(ETHZORCARacingLine, self).__init__(
            track=ETHZ(),
            car=ORCA(),
            nodes=[33, 67, 116, 166, 203, 239, 274, 309, 344, 362, 382, 407,
                   434, 448, 470, 514, 550, 586, 622, 657, 665],
            scale=0.95,
            lastidx=0,
            num_waypoints=100)


class ETHZMobilORCARacingLine(RacingLine):

    def __init__(self):

        super(ETHZMobilORCARacingLine, self).__init__(
            track=ETHZMobil(),
            car=ORCA(),
            nodes=[7, 21, 37, 52, 66, 81, 97, 111, 136, 160, 175, 191, 205,
                   220, 236, 250, 275, 299, 337, 376],
            scale=0.95,
            lastidx=0,
            num_waypoints=100)
