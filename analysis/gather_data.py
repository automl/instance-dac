from pathlib import Path
from instance_dac.utils.data_loading import load_eval_data

path = "../runs/Sigmoid"
train_instance_set = "2D3M_train"
train_instance_set_id = "sigmoid_2D3M_train"
agent_name = "ppo"
path = Path(path) / train_instance_set / agent_name
data = load_eval_data(path=path, train_instance_set_id=train_instance_set_id)