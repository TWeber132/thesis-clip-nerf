import json
import os
import time
import tensorflow as tf


import numpy as np
import wandb
from einops import rearrange
from loguru import logger
from manipulation_tasks import loader, factory

from lib.data_generator.util import camera_parameters
from lib.dataset.dataset import ColorDataset, MNPZDataset, NPZDataset, PickleDataset, SynchronizedDatasets
from lib.clip.utils import preprocess_tf
from lib.clip.utils import tokenize


def load_training_progress(eval_after_epochs, model_log_dir, n_epochs):
    start_epoch, training_progress_file = init_training_session(model_log_dir)
    start_n_fit = start_epoch // eval_after_epochs
    n_fits = n_epochs // eval_after_epochs
    best_mean_error = read_best_mean_error(training_progress_file)
    return best_mean_error, n_fits, start_epoch, start_n_fit, training_progress_file


def init_training_session(model_log_dir):
    start_epoch = 0
    training_progress_file = os.path.join(
        model_log_dir, 'training_progress.json')
    if os.path.exists(training_progress_file):
        with open(training_progress_file, 'r') as f:
            training_progress = json.load(f)
        if 'epoch' in training_progress:
            start_epoch = training_progress['epoch']
    logger.info('Starting training from epoch %d' % start_epoch)
    return start_epoch, training_progress_file


def read_best_mean_error(training_progress_file):
    best_mean_error = [2000, 2000]
    if os.path.exists(training_progress_file):
        with open(training_progress_file, 'r') as f:
            training_progress = json.load(f)
        if 'best_mean_error' in training_progress:
            best_mean_error = training_progress['best_mean_error']
    logger.info(f'Best mean error {best_mean_error}')
    return best_mean_error


def log_results(epoch, results, wandb_initialized):
    r_errors = [r['errors'] for r in results]
    mean_r_error = np.mean(np.concatenate(r_errors, axis=0), axis=0)

    best_errors_r = [errors_r[-1] for errors_r in r_errors]
    best_r_error_mean = np.mean(np.stack(best_errors_r, axis=0), axis=0)

    log_dict = {
        "epoch": epoch,
        "mean_r_error_t": mean_r_error[0] * 1000,
        "mean_r_error_r": mean_r_error[1] / np.pi * 180,
        "best_r_error_mean_t": best_r_error_mean[0] * 1000,
        "best_r_error_mean_r": best_r_error_mean[1] / np.pi * 180
    }

    logger.info(
        f"   Average   {log_dict['mean_r_error_t']}    {log_dict['mean_r_error_r']}")
    logger.info(
        f"   Best   {log_dict['best_r_error_mean_t']}    {log_dict['best_r_error_mean_r']}")
    if wandb_initialized:
        wandb.log(log_dict)


def get_inputs(dataset, sample_idx, n_images, grasp_model):
    observations = []
    intrinsics = []
    extrinsics_inv = []

    text = dataset.datasets['language'].read_sample(sample_idx)
    tokens = [tokenize(text)]  # 1 batch_size
    tokens = np.array(tokens, dtype=np.int32)

    if n_images == 2:
        for i in range(3, 5):
            src_img = dataset.datasets['color'].read_sample_at_idx(
                sample_idx, i)[..., :3] / 255.0
            camera_config = dataset.datasets['camera_config'].read_sample_at_idx(
                sample_idx, i)
            src_extr_inv, src_intr = camera_parameters(camera_config)
            observations.append(src_img)
            intrinsics.append(src_intr)
            extrinsics_inv.append(src_extr_inv)
    elif n_images == 3:
        for i in range(0, 3):
            src_img = dataset.datasets['color'].read_sample_at_idx(
                sample_idx, i)[..., :3] / 255.0
            camera_config = dataset.datasets['camera_config'].read_sample_at_idx(
                sample_idx, i)
            src_extr_inv, src_intr = camera_parameters(camera_config)
            observations.append(src_img)
            intrinsics.append(src_intr)
            extrinsics_inv.append(src_extr_inv)
    observations = np.array([observations])
    intrinsics = np.array([intrinsics])
    extrinsics_inv = np.array([extrinsics_inv])
    input_data = [observations.astype(np.float32),
                  intrinsics.astype(np.float32),
                  extrinsics_inv.astype(np.float32),
                  tokens]
    features = compute_features(input_data[0], tokens, grasp_model)
    task_info = dataset.datasets['info'].read_sample(sample_idx)
    return input_data, features, task_info


def compute_features(images, clip_tokens, grasp_model):
    src_images = rearrange(images, 'b n h w c -> (b n) h w c')

    clip_images = preprocess_tf(src_images)
    clip_visuals = grasp_model.clip_visual(clip_images)
    visual_features = grasp_model.visual_features(src_images)
    clip_textuals = grasp_model.clip_textual(clip_tokens)      # [(BN) 1024]
    combined_features = grasp_model.combine_clip_visual(
        (clip_visuals, visual_features, clip_textuals))

    combined_features = rearrange(
        combined_features, '(b n) h w c -> b n h w c', b=images.shape[0])
    return combined_features
