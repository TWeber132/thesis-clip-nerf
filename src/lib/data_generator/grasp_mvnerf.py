import numpy as np
from manipulation_tasks.transform import Affine

from lib.data_generator.base import DataGenerator
from lib.data_generator.util import camera_parameters


class GraspMVNeRFDataGenerator(DataGenerator):
    def __init__(self, dataset, workspace_bounds, n_views=1, n_points_train=512, batch_size=1, shuffle=True,
                 n_r_fraction=4):
        super(GraspMVNeRFDataGenerator, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.n_points_train = n_points_train
        self.n_negative = ((n_r_fraction - 1) * n_points_train) // n_r_fraction
        self.n_r_negative = self.n_points_train - self.n_negative - 1
        self.workspace_bounds = workspace_bounds
        self.n_views = n_views
        self.n_perspectives = self.dataset.datasets['color'].n_perspectives

    def get_data(self, batch):
        poses = []
        targets = []
        src_imagess = []
        src_intrinsicss = []
        src_extrinsics_invs = []
        for i in batch:
            if self.n_views == 1:
                src_indices = np.random.choice(range(3, 5), size=self.n_views, replace=False)
            elif self.n_views == 3:
                src_indices = np.random.choice(range(0, 3), size=self.n_views, replace=False)

            src_color = []
            src_extrinsics_inv = []
            src_intrinsics = []
            for src_index in src_indices:
                src_c = self.dataset.datasets['color'].read_sample_at_idx(i, src_index)[..., :3] / 255.0
                src_camera_config = self.dataset.datasets['camera_config'].read_sample_at_idx(i, src_index)
                src_color.append(src_c)
                src_extr_inv, src_intr = camera_parameters(src_camera_config)
                src_extrinsics_inv.append(src_extr_inv)
                src_intrinsics.append(src_intr)

            pose = self.dataset.datasets['grasp_pose'].read_sample(i)
            negative_samples = [Affine.random(self.workspace_bounds).matrix for _ in range(self.n_negative)]
            negative_r_transforms = [Affine.random(t_bounds=((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)),
                                                   allow_zero_rotation=False) for _ in range(self.n_r_negative)]
            negative_r_samples = [pose @ r_transform.matrix for r_transform in negative_r_transforms]
            all_poses = [pose, *negative_samples, *negative_r_samples]
            labels = np.concatenate((np.ones(1), np.zeros(self.n_points_train - 1)), axis=0)

            poses.append(all_poses)
            src_imagess.append(src_color)
            src_intrinsicss.append(src_intrinsics)
            src_extrinsics_invs.append(src_extrinsics_inv)

            targets.append(labels)

        inputs = [
            np.array(poses, dtype=np.float32),
            np.array(src_imagess, dtype=np.float32),
            np.array(src_intrinsicss, dtype=np.float32),
            np.array(src_extrinsics_invs, dtype=np.float32)
        ]
        return inputs, np.array(targets)
