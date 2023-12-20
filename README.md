# Instance-DAC

Runcommand
```bash
python instance_dac/train.py +benchmark=sigmoid +inst/sigmoid=2D3M_train
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default
```

Evaluate:
Override the test_set_path. Saved in train run dir under logs/eval/test_set_path.
Needs to have the same config as the train command.
```bash
python instance_dac/train.py +benchmark=sigmoid +inst/sigmoid=2D3M_train evaluate=True benchmark.config.test_set_path=../instance_sets/sigmoid/sigmoid_2D3M_test.csv
```

## Experiments
```bash
#####################################################
# SIGMOID
#####################################################
# 1. Train on 2D3M_train for 10 seeds
python instance_dac/train.py +benchmark=sigmoid +inst/sigmoid=2D3M_train 'seed=range(1,10)' -m

# Evaluate on train set
python instance_dac/train.py +benchmark=sigmoid +inst/sigmoid=2D3M_train evaluate=True benchmark.config.test_set_path=../instance_sets/sigmoid/sigmoid_2D3M_train.csv 'seed=range(1,10)' -m

# Evaluate on test set 1
python instance_dac/train.py +benchmark=sigmoid +inst/sigmoid=2D3M_train evaluate=True benchmark.config.test_set_path=../instance_sets/sigmoid/sigmoid_2D3M_test.csv 'seed=range(1,10)' -m


#####################################################
# CMA-ES
#####################################################
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default 'seed=range(1,21)' +cluster=noctua -m

```


## Installation
```bash
git clone https://github.com/automl/Instance-DAC.git
cd Instance-DAC
conda create -n instance_dac python=3.11
conda activate instance_dac

# Install for usage
pip install .

# Install for development
make install-dev


pip install -r requirements.txt
git clone git@github.com:automl/DACBench.git
cd DACBench
git checkout instance_dac
pip install -e .
```


## Data
With the script `instance_dac/collect_data.py` you can gather all the log files and create a csv file.
The table will be saved under `data/runs/<env_name>/<training_set>` and can contain performance data
on different test sets.
