from __future__ import annotations

# Generate instance set
import pandas as pd
from pathlib import Path
import ast


def generate_oracle_set(instance_set_path: str, instance_set_id: str, benchmark_id: str) -> tuple[str, int]:
    template = \
"""
# @package _global_

benchmark:  
    config:
        instance_set_path: {instance_set_path_train}
        test_set_path: ../instance_sets/sigmoid/sigmoid_2D3M_test.csv

instance_set_id: {instance_set_id}
"""

    # Assuming DACBench lies in instance-dac

    instance_set_path = Path("DACBench/dacbench/instance_sets") / instance_set_path

    benchmark_id_map = {
        "Sigmoid": "sigmoid"
    }

    benchmark_id_ = benchmark_id_map[benchmark_id]
    target_instance_set_config_dir = Path("instance_dac/configs/inst") / benchmark_id_ / f"oracle_{instance_set_id}"
    target_instance_set_dir = Path("../../../instance_dac/instance_sets") / benchmark_id_ / instance_set_id / "oracle"


    text = instance_set_path.read_text()
    lines = text.split("\n")
    n_instances = 0
    for line in lines:
        if line:
            n_instances += 1
            instance_id = ast.literal_eval(line)[0]
            instance_set_path_train = target_instance_set_dir / f"instance_{instance_id}.csv"
            instance_set_path_train.parent.mkdir(exist_ok=True, parents=True)
            instance_set_path_train.write_text(line + "\n")

            oracle_instance_set_id = f"{instance_set_id}_oracle_{instance_id}"
            config_text = template.format(instance_set_path_train=instance_set_path_train, instance_set_id=oracle_instance_set_id)
            config_path = target_instance_set_config_dir / f"instance_{instance_id}.yaml"
            config_path.parent.mkdir(exist_ok=True, parents=True)
            config_path.write_text(config_text)

    override = f"'+inst/{benchmark_id_}/oracle_{instance_set_id}=glob(*)'"

    return override, n_instances

if __name__ == "__main__":
    instance_set_path = "../instance_sets/sigmoid/sigmoid_2D3M_train.csv"
    instance_set_id = "2D3M_train"
    benchmark_id = "Sigmoid"

    override = generate_oracle_set(
        instance_set_path=instance_set_path,
        instance_set_id=instance_set_id,
        benchmark_id=benchmark_id,
    )
    print(override)

