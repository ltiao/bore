import pandas as pd

from hpbandster.core.worker import Worker
from bore.benchmarks import (Branin, StyblinskiTang, Michalewicz, Hartmann3D,
                             Hartmann6D, FCNet, FCNetAlt)

benchmarks = dict(
    branin=Branin,
    styblinski_tang=StyblinskiTang,
    michalewicz=Michalewicz,
    hartmann3d=Hartmann3D,
    hartmann6d=Hartmann6D,
    fcnet=FCNet,
    fcnet_alt=FCNetAlt
)


def make_name(benchmark_name, dimensions=None, dataset_name=None):

    if benchmark_name.startswith("fcnet"):
        assert dataset_name is not None, "must specify dataset name"
        name = f"fcnet_{dataset_name}"
    elif benchmark_name in ["michalewicz", "styblinski_tang"]:
        assert dimensions is not None, "must specify dimensions"
        name = f"{benchmark_name}_{dimensions:03d}d"
    else:
        name = benchmark_name

    return name


def make_benchmark(benchmark_name, dimensions=None, dataset_name=None,
                   input_dir=None):

    Benchmark = benchmarks[benchmark_name]

    kws = {}
    if benchmark_name.startswith("fcnet"):
        assert dataset_name is not None, "must specify dataset name"
        assert input_dir is not None, "must specify input directory"
        kws["dataset_name"] = dataset_name
        kws["data_dir"] = input_dir

    if benchmark_name in ["michalewicz", "styblinski_tang"]:
        assert dimensions is not None, "must specify dimensions"
        kws["dimensions"] = dimensions

    return Benchmark(**kws)


class BenchmarkWorker(Worker):

    def __init__(self, benchmark, *args, **kwargs):
        super(BenchmarkWorker, self).__init__(*args, **kwargs)
        self.benchmark = benchmark

    def compute(self, config, budget, **kwargs):
        evaluation = self.benchmark(config, budget)
        return dict(loss=evaluation.loss, info=evaluation.duration)


class HpBandSterLogs:

    def __init__(self, results):

        self.results = results

    def to_frame(self):

        rows = []

        for task, config_id in enumerate(self.results.data):

            d = self.results.data[config_id]
            bracket, _, _ = config_id

            for epoch in d.results:

                row = dict(task=task,
                           bracket=bracket,
                           epoch=int(epoch),
                           loss=d.results[epoch]["loss"],
                           info=d.results[epoch]["info"],
                           submitted=d.time_stamps[epoch]["submitted"],
                           started=d.time_stamps[epoch]["started"],
                           finished=d.time_stamps[epoch]["finished"])
                row.update(d.config)
                rows.append(row)

        return pd.DataFrame(rows)


class HyperOptLogs:

    def __init__(self, trials):
        self.trials = trials

    @staticmethod
    def get_value_item(dct, item):
        return {k: v[item] for k, v in dct.items()}

    def to_frame(self):

        t_start = None

        rows = []
        for trial in self.trials.trials:

            if t_start is None:
                t_start = trial["book_time"]

            started_timedelta = trial["book_time"] - t_start
            finished_timedelta = trial["refresh_time"] - t_start

            row = dict(task=trial["tid"],
                       started=started_timedelta.total_seconds(),
                       finished=finished_timedelta.total_seconds())
            # loss values, status, and other info
            row.update(trial["result"])
            # hyperparameter values
            row.update(self.get_value_item(trial["misc"]["vals"], item=0))
            rows.append(row)

        return pd.DataFrame(rows)
