from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dacbench.abstract_benchmark import AbstractBenchmark


def make_benchmark(cfg: DictConfig) -> AbstractBenchmark:
    benchmark = instantiate(cfg.benchmark)

    return benchmark