# RBMSolver
## Drift Control of High-Dimensional RBM: A Computational Method Based on Neural Networks
requirement: Numpy and Tensorflow 2

Usage:
```
python3 main.py --config_path=configs/RBMControlDP1drep.json 
```
*RBMControlDP1drep.json* can be replaced to other json file.

After running once, the data will be automatically dumped. Then, one can use the following commend
```
python3 main.py --config_path=configs/RBMControlDP1drep.json --dump=False
```

### Configs
TS or thin stream stands for linear cost structure.
DP or dynamic pricing stand for quadratic cost structure.

The code is adapted from https://github.com/MoZhou1995/DeepPDE_ActorCritic.