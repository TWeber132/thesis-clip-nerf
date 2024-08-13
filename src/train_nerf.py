import json
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig
from tensorflow_addons.optimizers import MultiOptimizer
import hydra

from src.lib.data_generator.mvnerf import MVNeRFDataGenerator
from src.lib.mvnerf.model_v0 import MVVNeRFRenderer, render_view
from src.lib.mvnerf.nerf_utils import WarmupScheduler, load_pretrained_weights
from src.lib.dataset.utils import load_dataset_nerf
from src.utils.util import init_training_session


def compile_model(nerf_renderer):
    mse = tf.keras.losses.MeanSquaredError()
    nerf_scheduler = WarmupScheduler(1e-4, 10000, 450000)
    optimizer_nerf = tf.keras.optimizers.Adam(learning_rate=nerf_scheduler)
    feature_scheduler = WarmupScheduler(1e-5, 10000, 450000)
    optimizer_feature = tf.keras.optimizers.Adam(
        learning_rate=feature_scheduler)
    optimizers_and_layers = [
        (optimizer_nerf, [nerf_renderer.coarse_embedding,
         nerf_renderer.fine_embedding]),
        (optimizer_feature,
         [nerf_renderer.visual_features.vision_transformer, nerf_renderer.visual_features.conv_features])
    ]
    optimizer = MultiOptimizer(optimizers_and_layers)
    nerf_renderer.compile(optimizer=optimizer, loss=mse)


def train_model(nerf_renderer, data_generator, n_epochs, eval_after_epochs, model_log_dir, model_checkpoint_name,
                valid_data):
    start_epoch, training_progress_file = init_training_session(model_log_dir)
    start_n_fit = start_epoch // eval_after_epochs
    n_fits = n_epochs // eval_after_epochs

    tgt_color = valid_data.pop('tgt_colors')

    if start_epoch == 0:
        combined_image = validate(nerf_renderer, tgt_color, valid_data)
        cv2.imwrite(
            f'{model_log_dir}/valid/valid-{start_epoch}.png', combined_image)
    for k in range(start_n_fit, n_fits):
        i_epoch = k * eval_after_epochs
        e_epoch = (k + 1) * eval_after_epochs
        nerf_renderer.fit(data_generator, epochs=e_epoch,
                          initial_epoch=i_epoch)

        combined_image = validate(nerf_renderer, tgt_color, valid_data)
        cv2.imwrite(
            f'{model_log_dir}/valid/valid-{e_epoch}.png', combined_image)

        # store training progress
        training_progress = {
            'epoch': e_epoch
        }
        with open(training_progress_file, 'w') as f:
            json.dump(training_progress, f)
        nerf_renderer.store(model_checkpoint_name)


def validate(nerf_renderer, tgt_color, valid_data):
    src_images = np.array([obs[..., :3] for obs in valid_data['src_colors']])
    combined_src_image = np.concatenate(src_images, axis=1)
    rendered_images = np.concatenate(
        (combined_src_image, tgt_color[..., :3]), axis=1)
    # resize rendered images to fit combined image
    rendered_image, rendered_depth = render_view(nerf_renderer, **valid_data)
    rendered_depth = cv2.cvtColor(rendered_depth, cv2.COLOR_GRAY2RGB)
    combined_image = np.concatenate((rendered_image, rendered_depth), axis=1)
    scale_factor = combined_image.shape[0] / rendered_images.shape[0]
    rendered_images = cv2.resize(rendered_images, (int(rendered_images.shape[1] * scale_factor),
                                                   int(rendered_images.shape[0] * scale_factor)))
    combined_image = np.concatenate((rendered_images, combined_image), axis=1)
    return combined_image


@hydra.main(version_base=None, config_path="configs", config_name="nerf_1_view")
def main(cfg: DictConfig) -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.run_functions_eagerly(True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    train_dataset = load_dataset_nerf(
        cfg.dataset.n_perspectives, cfg.dataset.path + '/train')
    valid_dataset = load_dataset_nerf(
        cfg.dataset.n_perspectives, cfg.dataset.path + '/valid')

    valid_data = {
        'src_colors': [valid_dataset.datasets['color'].read_sample_at_idx(cfg.valid_sample_idx, i) for i in
                       cfg.valid_perspective_src_indices[:cfg.nerf_model.n_views]],
        'src_camera_configs': [
            valid_dataset.datasets['camera_config'].read_sample_at_idx(
                cfg.valid_sample_idx, i)
            for i in cfg.valid_perspective_src_indices[:cfg.nerf_model.n_views]],
        'tgt_camera_config': valid_dataset.datasets['camera_config'].read_sample_at_idx(cfg.valid_sample_idx,
                                                                                        cfg.valid_perspective_tgt_idx),
        'tgt_colors': valid_dataset.datasets['color'].read_sample_at_idx(cfg.valid_sample_idx,
                                                                         cfg.valid_perspective_tgt_idx)
    }

    train_data_generator = MVNeRFDataGenerator(train_dataset,
                                               n_rays_train=cfg.nerf_model.n_rays_train,
                                               batch_size=cfg.nerf_training.batch_size,
                                               n_views=cfg.nerf_model.n_views,
                                               shuffle=True)

    nerf_renderer = MVVNeRFRenderer(**cfg.nerf_model,
                                    batch_size=cfg.nerf_training.batch_size)

    nerf_renderer(train_data_generator[0][0])
    compile_model(nerf_renderer)
    nerf_renderer.summary()
    nerf_renderer.combine_clip_visual.summary()

    os.makedirs(f'{cfg.nerf_training.model_path}/valid', exist_ok=True)
    model_checkpoint_name = f'{cfg.nerf_training.model_path}/model_final'

    if nerf_renderer.load(model_checkpoint_name):
        logger.info(f"Model loaded from {model_checkpoint_name}.")
    else:
        load_pretrained_weights(
            cfg.torch_weights_path, nerf_renderer.visual_features.vision_transformer)
        logger.info("New model initialized")

    train_model(nerf_renderer, train_data_generator, cfg.nerf_training.n_epochs,
                cfg.nerf_training.eval_after_epochs, cfg.nerf_training.model_path, model_checkpoint_name,
                valid_data)


if __name__ == '__main__':
    main()
