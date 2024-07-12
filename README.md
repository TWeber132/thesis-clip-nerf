# UR10e cell pybullet

# Quick start

checkout the run_dev.sh script and change the paths to fit your setup (DATA_HOST). Then run

```bash
./run_dev.sh
```

in the container you can generate data with

```bash
python src/train_nerf.py
```

this should start training and validation rendering after each epoch. The GPU has 16GB of memory, if your configuration
is different, adapt the batch size in `configs/training_condig/rtx3080_small.yaml`.

training_config is a config and is set via hydra. You can create your own configs in `src/configs`. check out
hydras documentation for more info at https://hydra.cc/docs/intro. the default config
is `src/configs/dataset_config.yaml`.

the validation images are stored in `DATA_HOST/models/models/nerf/simple/rtx3080_1v/valid`