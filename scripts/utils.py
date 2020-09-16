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
