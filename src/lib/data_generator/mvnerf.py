import numpy as np

from .base import DataGenerator
from .util import camera_parameters

from lib.mvnerf.nerf_utils import get_specific_rays, bbox_biased_sample


class MVNeRFDataGenerator(DataGenerator):
    def __init__(self, dataset, n_rays_train=512, batch_size=1, n_views=2, **kwargs):
        super().__init__(dataset, batch_size, **kwargs)
        self.n_rays_train = n_rays_train
        self.n_views = n_views
        self.n_perspectives = self.dataset.datasets['color'].n_perspectives

    def generate_rays(self, color, camera_config):
        tgt_extrinsic = camera_config['pose']
        tgt_intrinsic = np.reshape(camera_config['intrinsics'], (3, 3))
        tgt_intrinsic = tgt_intrinsic.astype(np.float32)
        rays = bbox_biased_sample(self.n_rays_train, np.array([0, 0, color.shape[0], color.shape[1]]),
                                  color.shape[0],
                                  color.shape[1])
        u, v = rays[:, 1], rays[:, 0]
        r_o, r_d = get_specific_rays(u, v, tgt_extrinsic, tgt_intrinsic)
        return r_d, r_o, rays

    @staticmethod
    def get_input(colors, camera_configs, r_d, r_o):
        src_extrinsic_invs = []
        src_intrinsics = []
        for camera_config in camera_configs:
            src_extrinsic_inv, src_intrinsic = camera_parameters(camera_config)
            src_extrinsic_invs.append(src_extrinsic_inv)
            src_intrinsics.append(src_intrinsic)

        inputs = (
            np.array([r_o], dtype=np.float32),
            np.array([r_d], dtype=np.float32),
            np.array([np.array(colors) / 255.0], dtype=np.float32),
            np.array([src_intrinsics], dtype=np.float32),
            np.array([src_extrinsic_invs], dtype=np.float32)
        )
        return inputs

    @staticmethod
    def get_target(color, rays):
        target_rgbs = np.array(color[rays[:, 0], rays[:, 1], :3]) / 255.0
        return target_rgbs

    def get_data(self, batch):
        rays_origins = []
        rays_directions = []
        src_images = []
        src_intrinsics = []
        src_extrinsics_inv = []
        targets = []
        for i in batch:
            indices = np.random.choice(
                range(self.n_perspectives), size=self.n_views + 1, replace=False)
            src_indices = indices[:-1]
            tgt_index = indices[-1]

            tgt_color = self.dataset.datasets['color'].read_sample_at_idx(
                i, tgt_index)[..., :3]
            tgt_camera_config = self.dataset.datasets['camera_config'].read_sample_at_idx(
                i, tgt_index)

            r_d, r_o, rays = self.generate_rays(tgt_color, tgt_camera_config)
            target_rgbs = MVNeRFDataGenerator.get_target(tgt_color, rays)

            src_colors = []
            src_camera_configs = []
            for src_index in src_indices:
                src_color = self.dataset.datasets['color'].read_sample_at_idx(
                    i, src_index)[..., :3]
                src_camera_config = self.dataset.datasets['camera_config'].read_sample_at_idx(
                    i, src_index)
                src_colors.append(src_color)
                src_camera_configs.append(src_camera_config)

            nn_input = MVNeRFDataGenerator.get_input(
                src_colors, src_camera_configs, r_d, r_o)

            rays_origins.extend(nn_input[0])
            rays_directions.extend(nn_input[1])

            src_images.extend(nn_input[2])
            src_intrinsics.extend(nn_input[3])
            src_extrinsics_inv.extend(nn_input[4])

            targets.append(target_rgbs)

        inputs = (
            np.array(rays_origins, dtype=np.float32),
            np.array(rays_directions, dtype=np.float32),
            np.array(src_images, dtype=np.float32),
            np.array(src_intrinsics, dtype=np.float32),
            np.array(src_extrinsics_inv, dtype=np.float32)
        )
        return inputs, np.array(targets)
