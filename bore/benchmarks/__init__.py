from .synthetic import (Ackley, Branin, GoldsteinPrice, Rosenbrock,
                        SixHumpCamel, StyblinskiTang, Michalewicz, Hartmann3D,
                        Hartmann6D)
from .tabular import FCNet, FCNetAlt
from .racing import (UCBF110RacingLine, ETHZORCARacingLine,
                     ETHZMobilORCARacingLine)

benchmarks = dict(
    branin=Branin,
    goldstein_price=GoldsteinPrice,
    six_hump_camel=SixHumpCamel,
    ackley=Ackley,
    rosenbrock=Rosenbrock,
    styblinski_tang=StyblinskiTang,
    michalewicz=Michalewicz,
    hartmann3d=Hartmann3D,
    hartmann6d=Hartmann6D,
    fcnet=FCNet,
    fcnet_alt=FCNetAlt,
    ucb_f110_racing=UCBF110RacingLine,
    ethz_orca_racing=ETHZORCARacingLine,
    ethz_mobil_orca_racing=ETHZMobilORCARacingLine
)


def make_benchmark(benchmark_name, dimensions=None, dataset_name=None,
                   data_dir=None):

    Benchmark = benchmarks[benchmark_name]

    kws = {}
    if benchmark_name.startswith("fcnet"):
        assert dataset_name is not None, "must specify dataset name"
        assert data_dir is not None, "must specify data directory"
        kws["dataset_name"] = dataset_name
        kws["data_dir"] = data_dir

    if benchmark_name in ("michalewicz", "styblinski_tang", "rosenbrock", "ackley"):
        assert dimensions is not None, "must specify dimensions"
        kws["dimensions"] = dimensions

    return Benchmark(**kws)
