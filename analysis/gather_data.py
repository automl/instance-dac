from pathlib import Path
from instance_dac.utils.data_loading import load_performance_data, load_eval_data

# LEVELS OF AGGREGATION
# 1. eval episode
# 2. seed
# 3. instance

# data.csv missing? Run
path = "../runs/Sigmoid"
agent_name = "ppo"
train_instance_set = "2D3M_train"
train_instance_set_id = "sigmoid_2D3M_train"
test_instance_set = "2D3M_test"
test_instance_set_id = "sigmoid_2D3M_test"
benchmark_id = "Sigmoid"

path = "../runs/CMA-ES"
agent_name = "ppo_sb3"
train_instance_set = "seplow_train"
test_instance_set_id = "test"
train_instance_set_id = "train"
test_instance_set = "seplow_test"
benchmark_id = "CMA-ES"

path = Path(path) / train_instance_set / agent_name
data = load_eval_data(path=path, instance_set_id=test_instance_set_id, instance_set=test_instance_set)