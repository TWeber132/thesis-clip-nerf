from util import get_inputs, get_outputs, get_info
from lib.simulation.environments.environment import Environment
from lib.simulation.tasks.picking_google_objects import PickingSeenGoogleObjectsSeq

task_names = {"picking-seen-google-objects-seq": PickingSeenGoogleObjectsSeq}


def validate_grasp_model(val_cfg, dataset, grasp_model):
    inputs = [get_inputs(dataset, i, int(
        val_cfg.grasp_opt_config.optimizer_config.n_images)) for i in val_cfg.valid_sample_indices]
    grasp_pose = grasp_model.infer(inputs)
    print(grasp_pose)
    expected_grasp_poses = [get_outputs(dataset, i)
                        for i in val_cfg.valid_sample_indices]
    print(expected_grasp_poses)

    # Calculate some score or visualize -> validate()
    # env.robot.gripper.closed()
    # env.robot.gripper.grasped_object(obj_uid)
    # z = p.GetBaseOrientationRotation(obj_uid)[0, 2]
    # z > 20cm

    # TODO: Get rid of split between data-generation config and nerf-training config


def visual_validation(cfg, dataset, act):

    task_info = [get_info(dataset, i)
                 for i in cfg.valid_sample_indices]
    task = task_names[cfg['task']](task_info)
    env = Environment(assets_root=cfg['assets_root'],
                      disp=cfg['disp'],
                      shared_memory=cfg['shared_memory'],
                      hz=480,
                      record_cfg=False)
    env.set_task(task)
    env.act(act)


def validate_user_input(val_cfg, dataset, grasp_model):
    user_input = input("Enter an object ot grasp: ")
    print(f"Entered: {user_input}")
    inputs = [get_inputs(dataset, i, int(
        val_cfg.grasp_opt_config.optimizer_config.n_images)) for i in val_cfg.valid_sample_indices]

    task = task_names[val_cfg['task']](task_info)
    task_info = [get_info(dataset, i)
                 for i in val_cfg.valid_sample_indices]
    env = Environment(assets_root=val_cfg['assets_root'],
                      disp=val_cfg['disp'],
                      shared_memory=val_cfg['shared_memory'],
                      hz=480,
                      record_cfg=False)
    env.set_task(task)
    act = grasp_model.infer(inputs)
    env.act(act)
