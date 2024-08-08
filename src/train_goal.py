import os
import sys

import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig
import hydra

from lib.data_generator.grasp_mvnerf import GraspMVNeRFDataGenerator
from lib.grasp_mvnerf.grasp_optimizer import GraspMVNeRFOptimizer
from lib.grasp_mvnerf.model import GraspMVNeRF
from lib.mvnerf.nerf_utils import load_pretrained_weights
from training import train_grasp_model
from util import load_dataset_baseline, setup_oracle, get_inputs

import wandb


@hydra.main(version_base=None, config_path="configs", config_name="goal_1_view")
def main(cfg: DictConfig) -> None:
    # allow memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    train_dataset = load_dataset_baseline(**cfg.dataset, dataset_type='train')
    valid_dataset = load_dataset_baseline(**cfg.dataset, dataset_type='valid')

    data_generator = GraspMVNeRFDataGenerator(train_dataset,
                                              **cfg.generator_grasp,
                                              batch_size=cfg.grasp_training.batch_size,
                                              n_views=cfg.nerf_model.n_views)

    softmax_before_loss = False
    if cfg.grasp_training.loss == 'kl_divergence':
        softmax_before_loss = True
    # initialize grasp model
    grasp_model = GraspMVNeRF(**cfg.grasp_model,
                              n_points_train=cfg.generator_grasp.n_points_train,
                              n_features=cfg.nerf_model.n_features,
                              original_image_size=list(cfg.nerf_model.original_image_size),
                              n_views=cfg.nerf_model.n_views,
                              softmax_before_loss=softmax_before_loss)
    _ = grasp_model(data_generator[0][0])

    if cfg.grasp_training.loss == 'cross_entropy':
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    elif cfg.grasp_training.loss == 'kl_divergence':
        loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    else:
        raise ValueError(f"Loss {cfg.grasp_training.loss} not supported.")

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.grasp_training.learning_rate)
    grasp_model.compile(optimizer=opt, loss=loss)

    backbone_checkpoint_name = os.path.join(cfg.grasp_training.backbone_path, 'model_final')
    if grasp_model.load_backbone(backbone_checkpoint_name):
        logger.info(f"Backbone loaded from {backbone_checkpoint_name}.")
    else:
        raise FileNotFoundError(f"Model not found at {backbone_checkpoint_name}.")

    os.makedirs(f'{cfg.grasp_training.model_path}/valid', exist_ok=True)
    model_checkpoint_name = f'{cfg.grasp_training.model_path}/model_final'
    if grasp_model.load(model_checkpoint_name):
        logger.info(f"Model loaded from {model_checkpoint_name}.")
    else:
        load_pretrained_weights(cfg.torch_weights_path, grasp_model.visual_features.vision_transformer)
        logger.info("New model initialized. Loaded pretrained weights.")

    grasp_optimizer = GraspMVNeRFOptimizer(grasp_model,
                                           **cfg.validation.grasp_opt_config.optimizer_config,
                                           workspace_bounds=cfg.generator_grasp.workspace_bounds)

    valid_data = [
        get_inputs(valid_dataset, i, int(cfg.validation.grasp_opt_config.optimizer_config.n_images), grasp_model) for i
        in cfg.valid_sample_indices]

    wandb_project_name = cfg.grasp_training.model_path.split('/')[-1]
    wandb_dir = f"{cfg.grasp_training.model_path}/wandb"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb_config = {
        'project': 'nerf-manipulation',
        'name': wandb_project_name,
        'dir': wandb_dir,
        'settings': wandb.Settings(start_method="fork")
    }

    validation_oracle = setup_oracle(cfg.validation.plugins, cfg.validation.oracle)
    train_grasp_model(grasp_model, data_generator, cfg.grasp_training.n_epochs,
                      cfg.grasp_training.eval_after_epochs, cfg.grasp_training.model_path, model_checkpoint_name,
                      grasp_optimizer, cfg.validation.grasp_opt_config.optimization_config, valid_data,
                      validation_oracle,
                      wandb_config)


if __name__ == '__main__':
    main()
