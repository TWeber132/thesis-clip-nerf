import os

import numpy as np
import tensorflow as tf
from einops import rearrange
from manipulation_tasks.transform import Affine
from tensorflow_addons.image import interpolate_bilinear

from ..mvnerf.nerf_utils import optimize

from ..grasp_mvnerf.layers import GraspReadout
from ..mvnerf.layers import VisualFeatures, MVResNetMLPNeRFEmbedding


class GraspMVNeRF(tf.keras.Model):
    def __init__(self, n_features=256, original_image_size=(480, 640), n_points_train=2048, n_views=1, n_5d_poses=7,
                 softmax_before_loss=False,
                 **kwargs):
        super(GraspMVNeRF, self).__init__(**kwargs)
        self.n_views = n_views

        self.fine_embedding = MVResNetMLPNeRFEmbedding(6, 128, n_views=n_views,
                                                       embed_direction_vector=True,
                                                       complete_output=True)

        self.visual_features = VisualFeatures(n_features, original_image_size)

        self.grasp_readout = GraspReadout(activation='elu')

        self.n_points_train = n_points_train

        self.n_5d_poses = n_5d_poses
        self.softmax_before_loss = softmax_before_loss
        base_offset_x = 0.02
        base_offset_y = 0.015
        base_offset_z = 0.0125
        step_size = (base_offset_x - 0.005) / ((self.n_5d_poses - 1) / 2)

        bases = [
            # t_base_1
            Affine(translation=[0, base_offset_y, 0]),
            # t_base_2
            Affine(translation=[0, -base_offset_y, 0]),
            # lf_base
            Affine(translation=[-base_offset_x, base_offset_y, base_offset_z], rotation=[0.0, np.pi / 2, 0.0]),
            # rf_base
            Affine(translation=[base_offset_x, base_offset_y, base_offset_z], rotation=[0.0, -np.pi / 2, 0.0]),
            # lb_base
            Affine(translation=[-base_offset_x, -base_offset_y, base_offset_z], rotation=[0.0, np.pi / 2, 0.0]),
            # # rb_base
            Affine(translation=[base_offset_x, -base_offset_y, base_offset_z], rotation=[0.0, -np.pi / 2, 0.0])
        ]
        centered_range = int((self.n_5d_poses - 1) / 2)
        transforms = [Affine(translation=[0.0, 0.0, i * step_size]) for i in
                      range(-centered_range, centered_range + 1)]

        # combine bases and transforms
        self.transforms_to_check_ = [tf.constant([(b * t).matrix], dtype=tf.float32) for b in bases
                                     for t in transforms]
        # convert to tensor
        self.transforms_to_check = tf.stack(self.transforms_to_check_, axis=0)
        self.transforms_to_check = self.transforms_to_check[tf.newaxis, ...]
        self.n_transforms_to_check = len(self.transforms_to_check_)
        self.z_dir = tf.constant([[[[[0], [0], [1]]]]], dtype=tf.float32)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def load_backbone(self, path, verbose=True):
        fine_embedding_path = f'{path}_fine_embedding'
        visual_features_path = f'{path}_visual_features'
        # check if <path>.index exists for all paths above
        if not os.path.exists(f'{fine_embedding_path}.index'):
            if verbose:
                print(f'{fine_embedding_path}.index does not exist')
            return False
        if not os.path.exists(f'{visual_features_path}.index'):
            if verbose:
                print(f'{visual_features_path}.index does not exist')
            return False

        self.fine_embedding.load_weights(fine_embedding_path)
        self.visual_features.load_weights(visual_features_path)
        return True

    def store(self, path):
        fine_embedding_path = f'{path}_fine_embedding'
        visual_features_path = f'{path}_visual_features'
        grasp_readout_path = f'{path}_grasp_readout'
        self.fine_embedding.save_weights(fine_embedding_path)
        self.visual_features.save_weights(visual_features_path)
        self.grasp_readout.save_weights(grasp_readout_path)

    def load(self, path):
        if self.load_backbone(path, verbose=False):
            grasp_readout_path = f'{path}_grasp_readout'
            if not os.path.exists(f'{grasp_readout_path}.index'):
                print(f'{grasp_readout_path}.index does not exist')
                return False
            self.grasp_readout.load_weights(f'{path}_grasp_readout')
            return True
        print(f'backbone models at {path} do not exist')
        return False

    def call(self, inputs, training=False, mask=None):
        src_images = inputs[1]
        src_images = rearrange(src_images, 'b n h w c -> (b n) h w c')
        batched_features = self.visual_features(src_images)
        batched_features = rearrange(batched_features, '(b n) h w c -> b n h w c', n=self.n_views)
        return self._call(inputs, self.n_points_train, batched_features)

    def infer(self, inputs, n_points_infer, batched_features):
        return self._call(inputs, n_points_infer, batched_features)

    def _call(self, inputs, n_points, batched_features):
        transforms = inputs[0]
        src_images = inputs[1]
        src_intrinsics = inputs[2]
        src_extrinsics_inv = inputs[3]

        normalized_images = src_images * 2 - 1.0
        transforms = transforms[:, tf.newaxis, ...]

        poses = tf.matmul(transforms, self.transforms_to_check)
        translations = poses[..., :3, 3]

        translations_homogeneous = tf.concat([translations, tf.ones_like(translations[..., :1])], axis=-1)
        translations_homogeneous = rearrange(translations_homogeneous, 'b n5 np d -> b () d (n5 np)')
        camera_points_homogeneous = tf.matmul(src_extrinsics_inv, translations_homogeneous)

        projections = tf.matmul(src_intrinsics, camera_points_homogeneous)
        pixel_locations = tf.math.divide(
            projections[..., :2, :], tf.maximum(projections[..., 2, tf.newaxis, :], 1e-8))
        pixel_locations = tf.clip_by_value(pixel_locations, clip_value_min=-1e6, clip_value_max=1e6)

        combined_features = tf.concat([normalized_images, batched_features], axis=-1)
        combined_features = rearrange(combined_features, 'b nv w h c -> (b nv) w h c')
        flat_pixel_locations = rearrange(pixel_locations, 'b nv d n5np -> (b nv) n5np d')
        combined_features = interpolate_bilinear(combined_features, flat_pixel_locations, indexing='xy')

        directions = tf.matmul(poses[..., :3, :3], self.z_dir)
        directions_homogeneous = tf.concat([directions, tf.ones_like(directions[..., :1, :])], axis=-2)
        directions_homogeneous = rearrange(directions_homogeneous, 'b n5 np d x -> b () d (n5 np x)')
        camera_directions = tf.matmul(src_extrinsics_inv, directions_homogeneous)[..., :3, :]

        camera_points = camera_points_homogeneous[..., :3, :]
        camera_points = rearrange(camera_points, 'b nv d (n5 np) -> (b nv) np n5 d',
                                  n5=self.n_transforms_to_check,
                                  np=n_points)
        camera_directions = rearrange(camera_directions, 'b nv d (n5 np) -> (b nv) np n5 d',
                                      n5=self.n_transforms_to_check,
                                      np=n_points)
        combined_features = rearrange(combined_features, 'bnv (n5 np) d -> bnv np n5 d',
                                      n5=self.n_transforms_to_check,
                                      np=n_points)

        embeddings = self.fine_embedding([camera_points, camera_directions, combined_features])[4:]
        grasp_success = self.grasp_readout(embeddings)

        return grasp_success

    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.grasp_readout.trainable_variables)
            energies = self(inputs, training=True)
            if self.softmax_before_loss:
                energies = tf.nn.softmax(energies)
            loss = self.loss(labels, energies)
        gradients = tape.gradient(loss, self.grasp_readout.trainable_variables)
        optimize(self.optimizer, self.grasp_readout.trainable_variables, gradients, 1.0)
        self.loss_tracker.update_state(loss)
        return {'loss': loss}
