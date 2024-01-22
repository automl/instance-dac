from __future__ import annotations

# Generate instance set
import pandas as pd
from pathlib import Path
import ast
from rich import print as printr


def generate_oracle_set(instance_set_path: str, instance_set_id: str, benchmark_id: str, test_set_path: str) -> tuple[str, int]:
    template = """
# @package _global_

benchmark:  
    config:
        instance_set_path: {instance_set_path_train}
        test_set_path: {test_set_path}

instance_set_id: {oracle_instance_set_id}
instance_id: {instance_id}
instance_set_selection: oracle
source_instance_set_id: {instance_set_id}
hydra:
  run:
    dir: runs/${{benchmark_id}}/${{source_instance_set_id}}/${{agent_name}}/${{instance_set_selection}}/instance_${{instance_id}}/${{seed}} 
  sweep:
    dir: runs/${{benchmark_id}}
    subdir: ${{source_instance_set_id}}/${{agent_name}}/${{instance_set_selection}}/instance_${{instance_id}}/${{seed}}
"""

    # Assuming DACBench lies in instance-dac

    instance_set_path = Path("DACBench/dacbench/instance_sets") / instance_set_path

    # Map benchmark id to name of yaml config file
    benchmark_id_map = {"Sigmoid": "sigmoid", "CMA-ES": "cmaes"}

    benchmark_id_ = benchmark_id_map[benchmark_id]
    target_instance_set_config_dir = Path("instance_dac/configs/inst") / benchmark_id_ / f"oracle_{instance_set_id}"
    target_instance_set_dir = Path("instance_dac/instance_sets") / benchmark_id_ / instance_set_id / "oracle"

    text = instance_set_path.read_text()
    lines = text.split("\n")
    n_instances = 0
    header_line = None
    if lines[0].startswith("ID"):
        header_line = lines[0]
    for line in lines:
        if line and not line.startswith("ID"):
            n_instances += 1
            instance_id = ast.literal_eval(line.split(",")[0])
            instance_set_path_train = target_instance_set_dir / f"instance_{instance_id}.csv"
            instance_set_path_train.parent.mkdir(exist_ok=True, parents=True)
            if header_line:
                line = header_line + "\n" + line
            instance_set_path_train.write_text(line + "\n")

            oracle_instance_set_id = f"{instance_set_id}_oracle_{instance_id}"
            config_text = template.format(
                instance_set_id=instance_set_id,
                instance_set_path_train="../../../" + str(instance_set_path_train),  # from the view of DACBench
                oracle_instance_set_id=oracle_instance_set_id,
                instance_id=str(instance_id),
                test_set_path=test_set_path,
            )
            config_path = target_instance_set_config_dir / f"instance_{instance_id}.yaml"
            config_path.parent.mkdir(exist_ok=True, parents=True)
            config_path.write_text(config_text)

    override = f"'+inst/{benchmark_id_}/oracle_{instance_set_id}=glob(*)'"

    return override, n_instances


if __name__ == "__main__":
    instance_set_path = "../instance_sets/sigmoid/sigmoid_2D3M_train.csv"
    instance_set_id = "2D3M_test"
    benchmark_id = "Sigmoid"
    test_set_path = "../instance_sets/sigmoid/sigmoid_2D3M_test.csv"

    override = generate_oracle_set(
        instance_set_path=instance_set_path,
        instance_set_id=instance_set_id,
        benchmark_id=benchmark_id,
        test_set_path=test_set_path
    )
    print(override)
