from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from rich import print as printr


def generate_instance_set(fname: str | Path):
    assert "Train" in str(fname)
    fname = Path(fname)

    print()
    print(fname)

    benchmark_id = fname.parts[2]
    instance_set_id = fname.parts[3]
    selected_on = fname.parts[4]
    selection_method = fname.parts[5]
    feature_type = fname.parts[6]
    source_features = fname.parts[7]
    threshold = fname.parts[8]
    seed = fname.parts[9]
    selector_run = fname.name.split("_")[-1].split(".")[0]

    if benchmark_id == "Sigmoid":
        test_set_path = "../instance_sets/sigmoid/sigmoid_2D3M_test.csv"
    else:
        test_set_path = "../../../instance_dac/instance_sets/cmaes/seplow/test.csv"

    source_instance_set = Path(
        f"DACBench/dacbench/instance_sets/{benchmark_id.lower()}/{benchmark_id.lower()}_{instance_set_id}.csv"
    )
    if not source_instance_set.is_file():
        source_instance_set = Path(
            f"instance_dac/instance_sets/{benchmark_id.lower().replace('-', '')}/seplow/train.csv"
        )
        if not source_instance_set.is_file():
            raise ValueError("Hardcoding did not work hehe", source_instance_set)

    iset = pd.read_csv(fname, header=None)
    iset = iset.dropna()

    instances = iset[1].values
    instance_ids = [int(i.split("_")[-1]) for i in instances]

    # TODO Discuss duplication!! When does it happen? Only for MIS?
    # assert len(set(instance_ids)) == len(instance_ids)
    instance_ids = list(set(instance_ids))

    with open(source_instance_set, "r") as file:
        lines = file.readlines()
    # lines = [l.strip("\n") for l in lines]
    offset = 0
    if benchmark_id == "CMA-ES":
        offset = 1
    filtered_set = [lines[i + 1] for i in instance_ids]
    if benchmark_id == "CMA-ES":
        filtered_set = [lines[0]] + filtered_set

    target_fname = (
        Path("data/instance_sets/selected/generated")
        / benchmark_id
        / instance_set_id
        / selected_on
        / selection_method
        / feature_type
        / source_features
        / threshold
        / seed
        / f"{selector_run}.csv"
    )
    target_fname.parent.mkdir(exist_ok=True, parents=True)
    with open(target_fname, "w") as file:
        file.writelines(filtered_set)
    print(target_fname)

    template = """
# @package _global_

benchmark:  
    config:
        instance_set_path: ../../../{instance_set_path_train}
        test_set_path: {test_set_path}

instance_set_id: {selector_instance_set_id}
instance_set_selection: selector
source_instance_set_id: {instance_set_id}
hydra:
  run:
    dir: runs/${{benchmark_id}}/${{source_instance_set_id}}/${{agent_name}}/${{instance_set_selection}}/${{instance_set_id}}/${{seed}} 
  sweep:
    dir: runs/${{benchmark_id}}
    subdir: ${{source_instance_set_id}}/${{agent_name}}/${{instance_set_selection}}/${{instance_set_id}}/${{seed}}

selector:
    graph_method: {selection_method}
    feature_type: {feature_type}
    feature_source: {source_features}
    threshold: {threshold}
    seed: {seed}
    run: {selector_run}
"""
    selector_instance_set_id = f"{instance_set_id}__{selected_on}__{selection_method}__{feature_type}__{source_features}__{threshold}__{seed}__{selector_run}"
    content = template.format(
        instance_set_path_train=target_fname,
        selector_instance_set_id=selector_instance_set_id,
        instance_set_id=instance_set_id,
        selection_method=selection_method,
        feature_type=feature_type,
        source_features=source_features,
        threshold=threshold,
        seed=seed,
        selector_run=selector_run,
        test_set_path=test_set_path,
    )

    name = selector_instance_set_id + ".yaml"
    if benchmark_id == "CMA-ES":
        benchmark_id = "cmaes"
    inst_config_fname = (
        Path(f"instance_dac/configs/inst/{benchmark_id.lower()}/selector") / f"source_{instance_set_id}" / name
    )
    inst_config_fname.parent.mkdir(exist_ok=True, parents=True)
    inst_config_fname.write_text(content)


if __name__ == "__main__":
    # fname = "data/selected_by_selector/Sigmoid/2D3M_train/Train/DS/Catch22/RA/0.7/dominant_0.7_1_use_params_False_sigmoid_2D3M_train_catch22.csv"
    path = Path("data/selected_by_selector/CMA-ES")
    fnames = list(path.glob("**/*.csv"))
    for fname in fnames:
        generate_instance_set(fname=fname)

    printr(f"Parsed {len(fnames)} files!")
