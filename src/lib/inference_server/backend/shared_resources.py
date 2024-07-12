import threading
import os

import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from loguru import logger

from lib.delta_ngf.grasp_optimizer import DNGFOptimizer
from lib.delta_ngf.model import DeltaNGF
from lib.grasp_mvnerf.grasp_optimizer import GraspMVNeRFOptimizer
from lib.grasp_mvnerf.model import GraspMVNeRF

_results = {}
_results_lock = threading.Lock()

_model_lock = threading.Lock()
_model_cache = {
    'model_name': None,
    'model': None
}


def set_result(task_id, value):
    with _results_lock:
        _results[task_id] = value


def get_result(task_id):
    with _results_lock:
        return _results.get(task_id, None)


def get_or_load_pose_optimizer(optimizer_name, init_poses=None, workspace_bounds=None):
    with _model_lock:
        reload_model = _model_cache['model_name'] != optimizer_name
        print(reload_model, _model_cache['model_name'], optimizer_name)
        if _model_cache['model'] is not None and init_poses is not None:
            if len(init_poses[0]) != _model_cache['model'].n_initial_guesses:
                reload_model = True
            print(len(init_poses[0]), _model_cache['model'].n_initial_guesses)
        if reload_model:
            pose_optimizer = load_optimizer(optimizer_name, init_poses, workspace_bounds)
            _model_cache['model_name'] = optimizer_name
            _model_cache['model'] = pose_optimizer
        else:
            logger.info(f"Model {optimizer_name} already loaded")
            pose_optimizer = _model_cache['model']
    return pose_optimizer


def remove_model_from_cache(model_name):
    with _model_lock:
        if _model_cache['model_name'] == model_name:
            _model_cache['model_name'] = None
            _model_cache['model'] = None


def load_optimizer(optimizer_name, init_poses=None, workspace_bounds=None):
    GlobalHydra.instance().clear()
    absolute_path = '/home/jovyan/data/config'
    current_path = os.getcwd()
    relative_path = os.path.relpath(absolute_path, start=current_path)
    initialize(config_path=relative_path, version_base=None)
    config_file_path = f'{absolute_path}/{optimizer_name}.yaml'
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file {config_file_path} not found.")
    cfg = compose(config_name=optimizer_name)
    print(OmegaConf.to_yaml(cfg))
    if init_poses is not None:
        cfg.optimizer_config.optimizer_config.n_initial_guesses = len(init_poses[0])

    optimizer_type = optimizer_name.split('_')[0]
    if optimizer_type == 'goal':
        # if optimizer_name in [
        #     'grasp_1_view_ce_quat',
        #     'grasp_1_view_ebm_quat',
        #     'grasp_1_view_ce_6d',
        #     'grasp_1_view_ebm_6d',
        #     'grasp_3_views'
        # ]:
        grasp_optimizer = load_optimizer_grasp(cfg, workspace_bounds)
    elif optimizer_type == 'trajectory':
        # elif optimizer_name in ['dngf_1_view_ebm_6d', 'dngf_1_view_ebm_quat',
        #                         'trajectory_1_view-1']:
        grasp_optimizer = load_optimizer_dngf(cfg, workspace_bounds)
    else:
        raise ValueError(f"Unknown optimizer name {optimizer_name}")
    logger.info(f"Model {optimizer_name} loaded with following config")
    print(OmegaConf.to_yaml(cfg))
    return grasp_optimizer


def load_optimizer_grasp(cfg, workspace_bounds=None):
    grasp_model = GraspMVNeRF(**cfg.grasp_model,
                              n_points_train=cfg.generator_grasp.n_points_train,
                              n_features=cfg.nerf_model.n_features,
                              original_image_size=list(cfg.nerf_model.original_image_size),
                              n_views=cfg.nerf_model.n_views)
    input_data = [
        np.zeros([1, cfg.generator_grasp.n_points_train, 4, 4]),
        np.zeros([1, cfg.nerf_model.n_views, *list(cfg.nerf_model.original_image_size), 3]),
        np.zeros([1, cfg.nerf_model.n_views, 4, 4]),
        np.zeros([1, cfg.nerf_model.n_views, 4, 4])
    ]
    _ = grasp_model(input_data)

    backbone_checkpoint_name = f'{cfg.backbone_path}/model_final'
    if not grasp_model.load_backbone(backbone_checkpoint_name):
        raise FileNotFoundError(f"Model not found at {backbone_checkpoint_name}.")
    model_checkpoint_name = f'{cfg.model_path}/model_final'
    if not grasp_model.load(model_checkpoint_name):
        raise FileNotFoundError(f"Model not found at {model_checkpoint_name}.")

    if workspace_bounds is None:
        workspace_bounds = cfg.generator_grasp.workspace_bounds
    grasp_optimizer = GraspMVNeRFOptimizer(grasp_model,
                                           **cfg.optimizer_config.optimizer_config,
                                           workspace_bounds=workspace_bounds)
    return grasp_optimizer


def load_optimizer_dngf(cfg, workspace_bounds=None):
    grasp_model = DeltaNGF(**cfg.grasp_model,
                           n_features=cfg.nerf_model.n_features,
                           n_views=cfg.nerf_model.n_views,
                           original_image_size=list(cfg.nerf_model.original_image_size),
                           n_points_train=cfg.generator_grasp.pose_augmentation_factor * cfg.generator_grasp.n_future_poses,
                           batch_size=1)
    input_data = [
        None,
        None,
        None,
        None,
        np.zeros([1, cfg.nerf_model.n_views, *list(cfg.nerf_model.original_image_size), 3]),
        np.zeros([1, cfg.nerf_model.n_views, 4, 4]),
        np.zeros([1, cfg.nerf_model.n_views, 4, 4])
    ]
    _ = grasp_model(input_data)

    backbone_checkpoint_name = f'{cfg.backbone_path}/model_final'
    if not grasp_model.load_backbone(backbone_checkpoint_name):
        raise FileNotFoundError(f"Model not found at {backbone_checkpoint_name}.")
    model_checkpoint_name = f'{cfg.model_path}/model_final'
    if not grasp_model.load(model_checkpoint_name):
        raise FileNotFoundError(f"Model not found at {model_checkpoint_name}.")

    if workspace_bounds is None:
        workspace_bounds = cfg.generator_grasp.workspace_bounds
    grasp_optimizer = DNGFOptimizer(grasp_model,
                                    **cfg.optimizer_config.optimizer_config,
                                    workspace_bounds=workspace_bounds)
    return grasp_optimizer
