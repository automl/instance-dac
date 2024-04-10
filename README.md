# Instance-DAC

Dynamic Algorithm Configuration (DAC) addresses the challenge of dynamically setting hyperparameters of an algorithm for a diverse set of instances rather than focusing solely on individual tasks.
Agents trained with Deep Reinforcement Learning (RL) offer a pathway to solve such settings.
However, the limited generalization performance of these agents has significantly hindered the application in DAC.
Our hypothesis is that a potential bias in the training instances limits generalization capabilities.
We take a step towards mitigating this by selecting a representative subset of training instances to overcome overrepresentation and then retraining the agent on this subset to improve its generalization performance. 
For constructing the meta-features for the subset selection, we particularly account for the dynamic nature of the RL agent by computing time series features on trajectories of actions and rewards generated by the agent's interaction with the environment.
Through empirical evaluations on the Sigmoid and CMA-ES benchmarks from the standard benchmark library for DAC, called DACBench, we discuss the potentials of our selection technique compared to training on the entire instance set.
Our results highlight the efficacy of instance selection in refining DAC policies for diverse instance spaces.

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
pip install -e .[modcma]


# SELECTOR
git clone https://github.com/gjorgjinac/InstanceDACSelector.git lib/InstanceDACSelector
pip install -r lib/InstanceDACSelector/requirements.txt
```

## Experiments
Check `scripts/sigmoid_experiment.sh` and `scripts/cmaes_experiment.sh` for commands/details how to run the experiments.

## Data
With the script `instance_dac/collect_data.py` you can gather all the log files and create a csv file.
The table will be saved under `data/runs/<env_name>/<training_set>` and can contain performance data
on different test sets.
