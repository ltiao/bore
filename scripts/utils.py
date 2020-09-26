import pandas as pd

from bore.benchmarks import (MichalewiczWorker, StyblinskiTangWorker,
                             Hartmann3DWorker, Hartmann6DWorker,
                             BoreholeWorker, FCNetWorker, BraninWorker)

workers = dict(
    # goldstein_price=GoldsteinPriceWorker,
    michalewicz=MichalewiczWorker,
    styblinski_tang=StyblinskiTangWorker,
    branin=BraninWorker,
    hartmann3d=Hartmann3DWorker,
    hartmann6d=Hartmann6DWorker,
    borehole=BoreholeWorker,
    fcnet=FCNetWorker
)


def get_worker(benchmark_name, dimensions=None, dataset_name=None,
               input_dir=None):

    Worker = workers.get(benchmark_name)

    kws = {}
    if benchmark_name == "fcnet":
        assert dataset_name is not None, "must specify dataset name"
        kws["dataset_name"] = dataset_name
        kws["data_dir"] = input_dir

    if benchmark_name in ["michalewicz", "styblinski_tang"]:
        assert dimensions is not None, "must specify dimensions"
        kws["dim"] = dimensions

    return Worker, kws


def get_name(benchmark_name, dimensions=None, dataset_name=None):

    if benchmark_name == "fcnet":
        assert dataset_name is not None, "must specify dataset name"
        name = f"{benchmark_name}_{dataset_name}"
    elif benchmark_name in ["michalewicz", "styblinski_tang"]:
        assert dimensions is not None, "must specify dimensions"
        name = f"{benchmark_name}_{dimensions:03d}d"
    else:
        name = benchmark_name

    return name


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
    def to_be_named(dct, item):

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
            row.update(self.to_be_named(trial["misc"]["vals"], item=0))
            rows.append(row)

        return pd.DataFrame(rows)
