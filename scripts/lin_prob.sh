python -m src.train experiment=lin_prob model=stl10 data=stl10 num_classes=10 pretrained_experiment=sim_clr_gpu_transforms trainer=gpu logger=wandb logger.wandb.name=lin_prob_sim_clr_stl10
python -m src.train experiment=lin_prob model=stl10 data=cifar10 num_classes=9 pretrained_experiment=sim_clr_gpu_transforms trainer=gpu logger=wandb logger.wandb.name=lin_prob_sim_clr_cifar10
python -m src.train experiment=lin_prob model=stl10 data=stl10 num_classes=10 pretrained_experiment=byol trainer=gpu logger=wandb logger.wandb.name=lin_prob_byol_stl10
python -m src.train experiment=lin_prob model=stl10 data=cifar10 num_classes=9 pretrained_experiment=byol trainer=gpu logger=wandb logger.wandb.name=lin_prob_byol_cifar10
