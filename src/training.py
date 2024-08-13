import json
import pickle

import numpy as np
import wandb
from loguru import logger

from util import load_training_progress, log_results
from optimization import validate


def init_wandb(wandb_config):
    run = None
    try:
        run = wandb.init(**wandb_config, resume=True)
        wandb_initialized = True
    except wandb.errors.UsageError:
        wandb_initialized = False
    logger.info(f"Wandb initialized: {wandb_initialized}")
    return run, wandb_initialized


def train_grasp_model(grasp_model, data_generator, n_epochs, eval_after_epochs, model_log_dir, model_checkpoint_name,
                      grasp_optimizer, optimization_config, wandb_config, valid_dataset):
    run, wandb_initialized = init_wandb(wandb_config)

    best_mean_error, n_fits, start_epoch, start_n_fit, training_progress_file = load_training_progress(
        eval_after_epochs, model_log_dir, n_epochs)

    # running the validation once before training is required for unknown reasons
    # otherwise an OOM error occurs during the first validation run
    # tried with and without memory growth, but it did not help
    # reinitializing the models did not help either
    # init_valid_len = len(valid_data) if start_epoch == 0 else 1
    valid_data = valid_dataset
    _ = validate(grasp_optimizer, optimization_config,
                 valid_data[:1])

    for k in range(start_n_fit, n_fits):
        i_epoch = k * eval_after_epochs
        e_epoch = (k + 1) * eval_after_epochs
        grasp_model.fit(data_generator, epochs=e_epoch, initial_epoch=i_epoch)

        results = validate(grasp_optimizer, optimization_config,
                           valid_data)
        valid_results_file = f'{model_log_dir}/valid/results-{e_epoch}.pkl'
        with open(valid_results_file, 'wb') as f:
            pickle.dump(results, f)

        log_results(e_epoch, results, wandb_initialized)

        r_errors = [r['errors_r'] for r in results]
        best_errors_final = [errors_r[-1] for errors_r in r_errors]
        new_mean_error = np.mean(np.stack(best_errors_final, axis=0), axis=0)

        if new_mean_error[0] * 1000 + new_mean_error[1] / np.pi * 180 < best_mean_error[0] * 1000 + best_mean_error[
                1] / np.pi * 180:
            grasp_model.store(f'{model_log_dir}/best')
            best_mean_error = list(new_mean_error)
            logger.info(
                f"New best mean error: {best_mean_error[0] * 1000}, {best_mean_error[1] / np.pi * 180}")

        # store training progress
        training_progress = {
            'epoch': e_epoch,
            'best_mean_error': best_mean_error
        }
        with open(training_progress_file, 'w') as f:
            json.dump(training_progress, f)
        grasp_model.store(model_checkpoint_name)
    if wandb_initialized:
        run.finish()
