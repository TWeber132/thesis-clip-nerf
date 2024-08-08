import os
import sys
import time

import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig
import hydra

from lib.data_generator.delta_ngf import DeltaNGFDataGenerator
from lib.delta_ngf.model import DeltaNGF
from lib.delta_ngf.grasp_optimizer import DNGFOptimizer
from lib.mvnerf.nerf_utils import load_pretrained_weights
from training import train_grasp_model
from util import load_dataset, load_dataset_baseline, setup_oracle, get_inputs

import wandb


@hydra.main(version_base=None, config_path="alt_configs", config_name="dngf_training_ebm_6d")
def main(cfg: DictConfig) -> None:
    # allow memory growth to avoid OOM errors
    # gpus = tf.config.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # eager execution
    # tf.config.run_functions_eagerly(True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if not cfg.dataset.record_grasp_pose:
        raise NotImplementedError
    train_dataset = load_dataset(**cfg.dataset, dataset_type='train')
    valid_dataset_cfg = dict(cfg.dataset)
    valid_dataset_cfg.pop('record_grasp_pose')
    valid_dataset_cfg.pop('record_order')
    valid_dataset = load_dataset_baseline(
        **valid_dataset_cfg, dataset_type='valid')

    data_generator = DeltaNGFDataGenerator(train_dataset,
                                           **cfg.generator_grasp,
                                           batch_size=cfg.grasp_training.batch_size,
                                           n_views=cfg.nerf_model.n_views,
                                           rotation_representation=cfg.grasp_model.rotation_representation)

    softmax_before_loss = False
    if cfg.grasp_training.loss == 'kl_divergence':
        softmax_before_loss = True
    # initialize grasp model
    grasp_model = DeltaNGF(**cfg.grasp_model,
                           n_features=cfg.nerf_model.n_features,
                           n_views=cfg.nerf_model.n_views,
                           original_image_size=list(
                               cfg.nerf_model.original_image_size),
                           batch_size=cfg.grasp_training.batch_size,
                           n_points_train=cfg.generator_grasp.pose_augmentation_factor *
                           cfg.generator_grasp.n_future_poses,
                           softmax_before_loss=softmax_before_loss)
    _ = grasp_model(data_generator[0][0])

    if cfg.grasp_training.loss == 'cross_entropy':
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    elif cfg.grasp_training.loss == 'kl_divergence':
        loss = tf.keras.losses.KLDivergence()
    else:
        raise ValueError(f"Loss {cfg.grasp_training.loss} not supported.")

    opt_landscape = tf.keras.optimizers.Adam(
        learning_rate=cfg.grasp_training.learning_rate)
    grasp_model.compile(optimizer=opt_landscape, loss=loss)

    backbone_checkpoint_name = os.path.join(
        cfg.grasp_training.backbone_path, 'model_final')
    if grasp_model.load_backbone(backbone_checkpoint_name):
        logger.info(f"Backbone loaded from {backbone_checkpoint_name}.")
    else:
        raise FileNotFoundError(
            f"Model not found at {backbone_checkpoint_name}.")

    os.makedirs(f'{cfg.grasp_training.model_path}/valid', exist_ok=True)
    model_checkpoint_name = f'{cfg.grasp_training.model_path}/model_final'
    if grasp_model.load(model_checkpoint_name):
        logger.info(f"Model loaded from {model_checkpoint_name}.")
    else:
        load_pretrained_weights(cfg.torch_weights_path,
                                grasp_model.visual_features.vision_transformer)
        logger.info("New model initialized. Loaded pretrained weights.")

    grasp_optimizer = DNGFOptimizer(grasp_model,
                                    **cfg.validation.grasp_opt_config.optimizer_config,
                                    workspace_bounds=cfg.generator_grasp.workspace_bounds,
                                    rotation_representation=cfg.grasp_model.rotation_representation)

    valid_data = [
        get_inputs(valid_dataset, i, int(cfg.validation.grasp_opt_config.optimizer_config.n_images), grasp_model) for i
        in cfg.valid_sample_indices]

    wandb_project_name = cfg.grasp_training.model_path.split('/')[-1]
    wandb_dir = f"{cfg.grasp_training.model_path}/wandb"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb_config = {
        'project': 'ras24',
        'name': wandb_project_name,
        'dir': wandb_dir,
        'settings': wandb.Settings(start_method="fork")
    }

    validation_oracle = setup_oracle(
        cfg.validation.plugins, cfg.validation.oracle)
    optimization_config = dict(
        cfg.validation.grasp_opt_config.optimization_config)
    optimization_config["sync"] = True
    train_grasp_model(grasp_model, data_generator, cfg.grasp_training.n_epochs,
                      cfg.grasp_training.eval_after_epochs, cfg.grasp_training.model_path, model_checkpoint_name,
                      grasp_optimizer, optimization_config, valid_data,
                      validation_oracle,
                      wandb_config)


if __name__ == '__main__':
    main()
