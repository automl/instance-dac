########
# Full training instance set
# Train
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default 'seed=range(1,11)' +cluster=noctua agent=ppo_sb3 -m
# Eval on test
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default 'seed=range(1,11)' +cluster=noctua agent=ppo_sb3 evaluate=True -m
# Eval on train
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default 'seed=range(1,11)' +cluster=noctua agent=ppo_sb3 evaluate=True eval_on_train_set=True -m

#########
# Train oracle
# Generate oracle configs
python instance_dac/main.py +benchmark=cmaes +inst/cmaes=default 'seed=range(1,11)' agent=ppo_sb3 -m --oracle --dry
# train on oracle configs
python instance_dac/train.py +benchmark=cmaes 'seed=range(1,11)' '+inst/cmaes/oracle_default=glob(*)' 'instance_set_selection=oracle' +cluster=noctua agent=ppo_sb3 -m
# Eval on oracle train (equals test set)
python instance_dac/train.py +benchmark=cmaes 'seed=range(1,11)' '+inst/cmaes/oracle_default=glob(*)' 'instance_set_selection=oracle' +cluster=noctua agent=ppo_sb3 evaluate=True eval_on_train_set=True -m

#########
# Eval random agent
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default evaluate=True agent=random 'seed=range(1,11)' -m


#########
# Random subsets
# Train on random subsets
python instance_dac/train.py +benchmark=cmaes 'seed=range(1,11)' '+inst/cmaes/random=glob(*)' 'instance_set_selection=random' +cluster=noctua agent=ppo_sb3 -m
# Eval trained on random subsets
python instance_dac/train.py +benchmark=cmaes 'seed=range(1,11)' '+inst/cmaes/random=glob(*)' 'instance_set_selection=random' +cluster=noctua agent=ppo_sb3 evaluate=True -m


###########
# SELECTOR
# Collect run data for SELECTOR
python -m instance_dac.collect_data.load_traineval_trajectories(path="runs/CMA-ES/seplow_train/ppo_sb3/full", train_instance_set_id="train")

# Calculate time series features
...

# Run SELECTOR (data is available)
python lib/instanceDACSElector/run_cmaes.py

# Preprocess output
...

# Train on subselected
python instance_dac/train.py +benchmark=cmaes 'seed=range(1,11)' '+inst/cmaes/selector=glob(*)' 'instance_set_selection=selector' +cluster=noctua agent=ppo_sb3 -m

# Eval on subselected
python instance_dac/train.py +benchmark=cmaes 'seed=range(1,11)' '+inst/cmaes/selector=glob(*)' 'instance_set_selection=selector' +cluster=noctua agent=ppo_sb3 evaluate=True -m

# analysis/oracle_distance.ipynb visualises the results as well as the other notebooks
