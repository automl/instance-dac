import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as printr


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    print("Hello world")
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)


if __name__ == "__main__":
    main()