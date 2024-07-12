from functools import reduce

import msgpack
import numpy as np

import tensorflow as tf

from loguru import logger
from manipulation_tasks.transform import Affine
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from shared_resources import remove_model_from_cache, set_result, get_or_load_pose_optimizer

from util import compute_features
from optimization import compute_results


def process_optimize_poses(byte_data, task_id):
    # byte_data, task_id = _optimize_poses_queue.get()

    data = msgpack.unpackb(byte_data, raw=False)

    optimizer_name = data["optimizer_name"]
    observations = data["observations"]
    optimization_config = data["optimization_config"]
    return_trajectory = data["return_trajectory"]
    if "init_poses" in data:
        init_poses = data["init_poses"]
        init_poses = preprocess_init_poses(init_poses)
    else:
        init_poses = None

    if "reset_optimizer" in data:
        reset_optimizer = data["reset_optimizer"]
    else:
        reset_optimizer = False

    if "clip_translation" in data:
        clip_translation = data["clip_translation"]
    else:
        clip_translation = False

    if "sync" in data:
        sync = data["sync"]
    else:
        sync = False

    if "workspace_bounds" in data:
        workspace_bounds = data["workspace_bounds"]
    else:
        workspace_bounds = None

    pose_optimizer = get_or_load_pose_optimizer(optimizer_name, init_poses, workspace_bounds)
    pose_optimizer.clip_translation = clip_translation

    input_data = preprocess_input(observations)
    features = compute_features(input_data[0], pose_optimizer.nerf_grasper.visual_features)

    try:
        _, losses_r, _, optimized_grasps_r, duration, all_poses = compute_results(pose_optimizer,
                                                                                  input_data, features,
                                                                                  return_trajectory, init_poses,
                                                                                  reset_optimizer,
                                                                                  **optimization_config,
                                                                                  sync=sync)
        logger.info(f"Optimization took {duration} seconds")
    except tf.errors.ResourceExhaustedError as e:
        remove_model_from_cache(optimizer_name)
        set_result(task_id, {"status": "failed", "message": "Server GPU memory exhausted",
                             "error_message": str(e)})
        return
    except InvalidArgumentError as e:
        set_result(task_id, {"status": "failed", "message": "Invalid argument",
                             "error_message": str(e)})
        return

    best_grasp_indices = np.argsort(losses_r)
    current_i = len(best_grasp_indices) - 1
    downwards = False
    while not downwards:
        best_grasp_index = best_grasp_indices[current_i]
        best_grasp_pose = optimized_grasps_r[best_grasp_index]
        best_grasp_loss = losses_r[best_grasp_index]
        z_axis = best_grasp_pose.matrix[:3, 2]
        cos_angle = np.dot(z_axis, np.array([0, 0, 1]))
        if cos_angle < 0:
            downwards = True
        else:
            current_i -= 1

    # best_grasp_pose = optimized_grasps_r[best_grasp_index]
    best_grasp_message = best_grasp_pose.matrix.astype(np.float32).tobytes()
    trajectory = [poses[best_grasp_index] for poses in all_poses]
    trajectory_message = [pose.matrix.astype(np.float32).tobytes() for pose in trajectory]
    set_result(task_id, {"status": "completed",
                         "optimized_pose": best_grasp_message,
                         "optimized_loss": float(best_grasp_loss),
                         "trajectory": trajectory_message,
                         "duration": duration})


def process_generate_trajectories(byte_data, task_id):
    # byte_data, task_id = _optimize_poses_queue.get()
    data = msgpack.unpackb(byte_data, raw=False)

    optimizer_name = data["optimizer_name"]
    observations = data["observations"]
    optimization_config = data["optimization_config"]
    return_trajectory = data["return_trajectory"]
    if "init_poses" in data:
        init_poses = data["init_poses"]
        init_poses = preprocess_init_poses(init_poses)
    else:
        init_poses = None

    if "reset_optimizer" in data:
        reset_optimizer = data["reset_optimizer"]
    else:
        reset_optimizer = False

    if "clip_translation" in data:
        clip_translation = data["clip_translation"]
    else:
        clip_translation = False

    if "sync" in data:
        sync = data["sync"]
    else:
        sync = False
    
    if "workspace_bounds" in data:
        workspace_bounds = data["workspace_bounds"]
    else:
        workspace_bounds = None

    pose_optimizer = get_or_load_pose_optimizer(optimizer_name, init_poses, workspace_bounds)
    pose_optimizer.clip_translation = clip_translation

    input_data = preprocess_input(observations)
    features = compute_features(input_data[0], pose_optimizer.nerf_grasper.visual_features)

    try:
        _, losses_r, _, optimized_grasps_r, duration, all_poses = compute_results(pose_optimizer,
                                                                                  input_data, features,
                                                                                  return_trajectory, init_poses,
                                                                                  reset_optimizer,
                                                                                  **optimization_config,
                                                                                  sync=sync)
        logger.info(f"Optimization took {duration} seconds")
    except tf.errors.ResourceExhaustedError as e:
        remove_model_from_cache(optimizer_name)
        set_result(task_id, {"status": "failed", "message": "Server GPU memory exhausted",
                             "error_message": str(e)})
        return
    except InvalidArgumentError as e:
        set_result(task_id, {"status": "failed", "message": "Invalid argument",
                             "error_message": str(e)})
        return

    all_poses_message = [[pose.matrix.astype(np.float32).tobytes() for pose in poses] for poses in all_poses]
    all_losses_message = losses_r.astype(np.float32).tobytes()
    all_optimized_grasps_message = [pose.matrix.astype(np.float32).tobytes() for pose in optimized_grasps_r]
    set_result(task_id, {"status": "completed",
                         "all_poses": all_poses_message,
                         "all_losses": all_losses_message,
                         "all_optimized_grasps": all_optimized_grasps_message,
                         "duration": duration})




def preprocess_input(observations):
    src_images = []
    src_intrinsics = []
    src_extrinsics_inv = []
    for o in observations:
        src_image = o['color']
        src_image_shape = o['color_shape']
        src_image = np.frombuffer(src_image, dtype=np.uint8).reshape(src_image_shape)[..., :3] / 255.0

        src_intrinsic = o['intrinsics']
        src_intrinsic = np.frombuffer(src_intrinsic, dtype=np.float32).reshape(3, 3)
        src_intrinsic = np.concatenate([src_intrinsic, np.zeros((3, 1))], axis=1)
        src_intrinsic = np.concatenate([src_intrinsic, np.array([[0, 0, 0, 1]])], axis=0)

        src_extrinsic = o['extrinsics']
        src_extrinsic = np.frombuffer(src_extrinsic, dtype=np.float32).reshape(4, 4)
        src_extrinsic_inv = np.linalg.inv(src_extrinsic)

        src_images.append(src_image)
        src_intrinsics.append(src_intrinsic)
        src_extrinsics_inv.append(src_extrinsic_inv)
    input_data = [np.array([src_images]).astype(np.float32),
                  np.array([src_intrinsics]).astype(np.float32),
                  np.array([src_extrinsics_inv]).astype(np.float32)]
    return input_data


def preprocess_init_poses(init_poses):
    init_poses = [np.frombuffer(pose, dtype=np.float32).reshape(4, 4) for pose in init_poses]
    init_poses = [Affine.from_matrix(pose) for pose in init_poses]
    init_translations = [pose.translation for pose in init_poses]
    init_rotations = [pose.quat for pose in init_poses]
    return [np.array([init_translations]).astype(np.float32),
            np.array([init_rotations]).astype(np.float32)]
