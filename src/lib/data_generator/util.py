import numpy as np


def camera_parameters(camera_config):
    src_extrinsic = camera_config['pose']
    src_intrinsic = np.reshape(camera_config['intrinsics'], (3, 3))
    src_intrinsic = np.concatenate((src_intrinsic, np.zeros((3, 1))), axis=1)
    src_intrinsic = np.concatenate((src_intrinsic, np.array([[0, 0, 0, 1]])), axis=0)
    src_extrinsic_inv = np.linalg.inv(src_extrinsic)
    return src_extrinsic_inv, src_intrinsic
