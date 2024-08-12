import time
import numpy as np
import tensorflow as tf
from loguru import logger
from transforms3d import quaternions


from lib.agents.oracle_agent import OracleAgent



def validate(pose_optimizer, optimization_config, valid_data):
    results = []
    durations = []
    best_errors_r = []
    all_errors_r = []
    for i, (input_data, features, _task_info, grasp_pose_h) in enumerate(valid_data):
        logger.info(
            f"Validating on sample {i + 1} with {1.232} objects ...")
        losses_t, losses_r, optimized_grasps_t, optimized_grasps_r, duration, _ = compute_results(
            pose_optimizer,
            input_data, features, False,
            **optimization_config)

        result = get_step_results(
            losses_t, losses_r, optimized_grasps_t, optimized_grasps_r, grasp_pose_h)
        results.append(result)
        durations.append(duration)
        errors_r = result['errors_r']
        all_errors_r.append(errors_r)

        best_error_r_idx = -1
        best_error_r = errors_r[best_error_r_idx]
        logger.info(
            f"   Best    {best_error_r[0] * 1000}    {best_error_r[1] / np.pi * 180}")

        best_errors_r.append(best_error_r)
    return results


def compute_results(pose_optimizer, input_data, features, return_trajectory, init_poses=None, reset_optimizer=True,
                    n_optimization_steps=1,
                    init_lr_t=0.09, decay_t=None, init_lr_r=None, decay_r=None, sync=False):
    if reset_optimizer:
        if init_lr_r is None:
            init_lr_r = init_lr_t
        if decay_r is None:
            decay_r = decay_t

        lr_schedule_t = tf.keras.optimizers.schedules.ExponentialDecay(
            init_lr_t,
            decay_steps=1,
            decay_rate=decay_t,
            staircase=False)
        optimizer_t = tf.keras.optimizers.Adam(learning_rate=lr_schedule_t)

        lr_schedule_q = tf.keras.optimizers.schedules.ExponentialDecay(
            init_lr_r,
            decay_steps=1,
            decay_rate=decay_r,
            staircase=False)
        optimizer_q = tf.keras.optimizers.Adam(learning_rate=lr_schedule_q)

        optimizers = [optimizer_t, optimizer_q]
        pose_optimizer.compile(optimizer=optimizers)

    if init_poses is not None:
        pose_optimizer.set_initial_guesses(init_poses)
    else:
        init_guesses = pose_optimizer.generate_initial_guesses()
        pose_optimizer.set_initial_guesses(init_guesses)

    duration = 0
    if not isinstance(n_optimization_steps, list):
        _n_optimization_steps = [n_optimization_steps]
    else:
        _n_optimization_steps = n_optimization_steps
    all_poses = []
    if return_trajectory:
        poses = pose_optimizer.get_results()
        all_poses.append(poses)
    for o_steps in _n_optimization_steps:
        if not sync:
            train_config = [True, False]
            optimized_grasps_t, losses_t, duration_t, poses = optimize_pose(pose_optimizer, input_data, features,
                                                                            train_config, o_steps, return_trajectory)
            if return_trajectory:
                all_poses.extend(poses)
            train_config = [False, True]
            optimized_grasps_r, losses_r, duration_r, poses = optimize_pose(pose_optimizer, input_data, features,
                                                                            train_config, o_steps, return_trajectory)
            if return_trajectory:
                all_poses.extend(poses)

            duration += duration_t + duration_r
        else:
            train_config = [True, True]
            optimized_grasps_r, losses_r, duration, poses = optimize_pose(pose_optimizer, input_data, features,
                                                                          train_config, o_steps, return_trajectory)
            losses_t = losses_r
            optimized_grasps_t = optimized_grasps_r
            if return_trajectory:
                all_poses.extend(poses)
            duration += duration

    return losses_t, losses_r, optimized_grasps_t, optimized_grasps_r, duration, all_poses


def get_step_results(losses_t, losses_r, trajectory_t, trajectory_r, gt_grasp_pose_h):
    oracle = OracleAgent()
    quat = quaternions.mat2quat(gt_grasp_pose_h[:3, :3])
    quat = quat[[1, 2, 3, 0]]
    gt_grasp_pose = [tuple([*gt_grasp_pose_h[:3, 3]]),tuple([*quat])]
    gt_action = [gt_grasp_pose]
    
    # determine the best 5 grasp indices based on their final success
    # best_grasp_indices_t = np.argsort(losses_t)[-5:]
    best_grasp_indices_r = np.argsort(losses_r)[-5:]

    # get the best 5 grasp poses
    best_grasp_poses_r = [trajectory_r[k] for k in best_grasp_indices_r]
    final_success_r = [losses_r[k] for k in best_grasp_indices_r]
    errors_r = []
    for k in range(len(best_grasp_poses_r)):
        print(best_grasp_poses_r[k])
        best_pose = [tuple([*best_grasp_poses_r[k].translation]), tuple([*best_grasp_poses_r[k].quat])]
        best_action = [best_pose]

        trans_error, rot_error = oracle.calculate_errors(
            gt_action, best_action)
        # t_error, r_error = oracle.calculate_errors(gt_action, action)
        print(trans_error, rot_error)
        errors_r.append([trans_error, rot_error])

    results = {
        'grasp_poses': best_grasp_poses_r,
        'final_success': final_success_r,
        'errors_r': errors_r
    }
    return results


def optimize_pose(pose_optimizer, input_data, batched_features, train_config, n_optimization_steps=16,
                  return_trajectory=False):
    start = time.time()
    step_poses = []
    for j in range(n_optimization_steps):
        ret = pose_optimizer.optimize_pose(
            input_data, batched_features, train_config=train_config)
        poses = []
        if return_trajectory:
            poses = pose_optimizer.get_results()
        step_poses.append(poses)
    optimized_grasps = pose_optimizer.get_results()
    step_poses.append(optimized_grasps)
    losses = pose_optimizer.compute_current_grasp_success(
        input_data, batched_features).numpy().squeeze()
    end = time.time()
    duration = end - start
    return optimized_grasps, losses, duration, step_poses
