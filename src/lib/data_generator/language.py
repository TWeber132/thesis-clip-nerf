import cv2
import numpy as np
from manipulation_tasks.transform import Affine

from lib.data_generator.base import DataGenerator
from lib.data_generator.util import camera_parameters


class LanguageDataGenerator(DataGenerator):
    def __init__(self, dataset, workspace_bounds, n_views=1, batch_size=1, shuffle=True,
                 pose_augmentation_factor=1, n_future_poses=5, fixed_orientation=None,
                 rotation_representation='quaternion'):
        self.future_poses = n_future_poses
        self.pose_augmentation_factor = pose_augmentation_factor
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle)
        self.workspace_bounds = workspace_bounds
        self.n_views = n_views
        self.n_perspectives = self.dataset.datasets['color'].n_perspectives

        # self.fixed_orientation = [np.pi, 0.0, np.pi / 2]
        self.fixed_orientation = fixed_orientation
        self.rotation_representation = rotation_representation

        self.n_points_train = self.future_poses * self.pose_augmentation_factor
        if self.fixed_orientation is not None:
            self.n_negative = self.n_points_train - self.future_poses
            self.n_r_negative = 0
        else:
            n_r_fraction = 8
            self.n_negative = (
                (n_r_fraction - 1) * self.n_points_train) // n_r_fraction - self.future_poses
            self.n_r_negative = self.n_points_train - self.n_negative - self.future_poses

    def get_data_camera(self, batch):
        src_imagess = []
        src_intrinsicss = []
        src_extrinsics_invs = []

        for i in batch:
            src_indices = np.random.choice(
                range(self.n_perspectives), size=self.n_views, replace=False)
            src_color = []
            src_extrinsics_inv = []
            src_intrinsics = []
            for src_index in src_indices:
                src_c = self.dataset.datasets['color'].read_sample_at_idx(
                    i, src_index)[..., :3] / 255.0
                src_camera_config = self.dataset.datasets['camera_config'].read_sample_at_idx(
                    i, src_index)
                src_color.append(src_c)
                src_extr_inv, src_intr = camera_parameters(src_camera_config)
                src_extrinsics_inv.append(src_extr_inv)
                src_intrinsics.append(src_intr)
            src_imagess.append(src_color)
            src_intrinsicss.append(src_intrinsics)
            src_extrinsics_invs.append(src_extrinsics_inv)

        src_imagess = np.array(src_imagess, dtype=np.float32)
        src_intrinsicss = np.array(src_intrinsicss, dtype=np.float32)
        src_extrinsics_invs = np.array(src_extrinsics_invs, dtype=np.float32)
        return src_imagess, src_intrinsicss, src_extrinsics_invs

    def get_data_landscape_final(self, batch):
        input_translations = []
        input_rotations = []
        targets = []

        for i in batch:
            target_pose = self.dataset.datasets['grasp_pose'].read_sample(i)

            negative_samples = [Affine.random(self.workspace_bounds).matrix for _ in
                                range(self.n_negative + self.future_poses - 1)]

            negative_r_transforms = [Affine.random(t_bounds=((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)),
                                                   allow_zero_rotation=False) for _ in range(self.n_r_negative)]
            negative_r_samples = [
                target_pose @ r_transform.matrix for r_transform in negative_r_transforms]
            all_poses = [target_pose, *negative_samples, *negative_r_samples]
            labels = np.concatenate((np.ones(1), np.zeros(self.n_points_train - 1)),
                                    axis=0)
            input_translation = [Affine.from_matrix(
                pose).translation for pose in all_poses]
            input_translations.append(input_translation)
            if self.rotation_representation == 'quaternion':
                input_rotation = [Affine.from_matrix(
                    pose).quat for pose in all_poses]
            elif self.rotation_representation == '6d':
                input_rotation = [
                    np.concatenate([Affine.from_matrix(
                        pose).rotation[:, 0], Affine.from_matrix(pose).rotation[:, 1]])
                    for pose in all_poses]
            input_rotations.append(input_rotation)
            targets.append(labels)
        input_translations = np.array(input_translations, dtype=np.float32)
        input_rotations = np.array(input_rotations, dtype=np.float32)
        targets = np.array(targets)
        return input_translations, input_rotations, targets

    def get_data_grad(self, batch):
        translations = []
        rotations = []
        target_d_t = []
        target_d_q = []

        for i in batch:
            trajectory = self.dataset.datasets['trajectory'].read_sample(i)
            initial_index = np.random.randint(
                0, len(trajectory) - self.future_poses - 1)
            required_poses = trajectory[initial_index:
                                        initial_index + self.future_poses + 1]

            augmented_poses = []
            augmented_targets = []
            for j, pose in enumerate(required_poses[:-1]):
                for _ in range(self.pose_augmentation_factor):
                    augmentation = Affine.random(t_bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02)),
                                                 r_bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)))
                    input_pose = pose @ augmentation.matrix
                    target_pose = required_poses[j + 1]
                    if self.fixed_orientation is not None:
                        input_pose = Affine(translation=Affine.from_matrix(input_pose).translation,
                                            rotation=self.fixed_orientation).matrix
                        target_pose = Affine(translation=Affine.from_matrix(target_pose).translation,
                                             rotation=self.fixed_orientation).matrix

                    augmented_poses.append(input_pose)
                    augmented_targets.append(target_pose)

            input_translations = [Affine.from_matrix(
                pose).translation for pose in augmented_poses]
            target_translations = [Affine.from_matrix(
                pose).translation for pose in augmented_targets]
            if self.rotation_representation == 'quaternion':
                input_rotations = [Affine.from_matrix(
                    pose).quat for pose in augmented_poses]
                target_rotations = [Affine.from_matrix(
                    pose).quat for pose in augmented_targets]
            elif self.rotation_representation == '6d':
                input_rotations = [
                    np.concatenate([Affine.from_matrix(
                        pose).rotation[:, 0], Affine.from_matrix(pose).rotation[:, 1]])
                    for pose in augmented_poses]
                target_rotations = [
                    np.concatenate([Affine.from_matrix(
                        pose).rotation[:, 0], Affine.from_matrix(pose).rotation[:, 1]])
                    for pose in augmented_targets]

            d_translations = [target_translation - input_translation for target_translation, input_translation in
                              zip(target_translations, input_translations)]
            d_quaternions = [target_quaternion - input_quaternion for target_quaternion, input_quaternion in
                             zip(target_rotations, input_rotations)]
            translations.append(input_translations)
            rotations.append(input_rotations)
            target_d_t.append(d_translations)
            target_d_q.append(d_quaternions)
        translations = np.array(translations, dtype=np.float32)
        rotations = np.array(rotations, dtype=np.float32)
        target_d_t = np.array(target_d_t, dtype=np.float32)
        target_d_q = np.array(target_d_q, dtype=np.float32)
        return translations, rotations, target_d_t, target_d_q

    def get_data_text(self, batch):
        languages = []
        for i in batch:
            language = self.dataset.datasets['language'].read_sample(i)
            languages.append(language)
        return languages

    def get_data(self, batch):
        src_imagess, src_intrinsicss, src_extrinsics_invs = self.get_data_camera(
            batch)
        input_translations, input_rotations, targets = self.get_data_landscape_final(
            batch)
        translations, rotations, target_d_t, target_d_r = self.get_data_grad(
            batch)
        texts = self.get_data_text(batch)
        inputs = [
            input_translations,
            input_rotations,
            translations,
            rotations,
            src_imagess,
            src_intrinsicss,
            src_extrinsics_invs,
            texts
        ]
        targets = [
            targets,
            target_d_t,
            target_d_r
        ]
        return inputs, targets
