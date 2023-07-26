# Instance-DAC

Runcommand
```bash
python instance_dac/train.py +benchmark=sigmoid_easy
```

Evaluate:
Override the instance_set_path. Saved in train run dir under logs/eval/instance_set_path.
```bash
python instance_dac/train.py +benchmark=sigmoid_easy evaluate=True benchmark.config.instance_set_path=../instance_sets/sigmoid/sigmoid_2D3M_test.csv
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

Documentation at https://automl.github.io/Instance-DAC/main

## Minimal Example

```
# Your code here
```
