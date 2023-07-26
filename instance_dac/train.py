import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from instance_dac.make import make_benchmark


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    benchmark = make_benchmark(cfg=cfg)
    env = benchmark.get_environment()


if __name__ == "__main__":
    main()