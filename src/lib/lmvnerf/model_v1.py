import os

import numpy as np
import tensorflow as tf
from einops import rearrange
from manipulation_tasks.transform import Affine
from tensorflow_addons.image import interpolate_bilinear
import tensorflow_graphics.geometry.transformation as tf_transformation

from ..mvnerf.nerf_utils import optimize

from ..delta_ngf.layers import GraspReadout
from ..mvnerf.layers import VisualFeatures, MVResNetMLPNeRFEmbedding, CombineCLIPVisualV1
from ..clip.model import CLIPVisualEncoder, CLIPTextualEncoder
from ..clip.utils import preprocess_tf


def t_m_to_h_matrix(translations, rot_matrices):
    batch_size = tf.shape(rot_matrices)[0]
    n_points = tf.shape(rot_matrices)[1]
    last_row = tf.tile(tf.constant([[0., 0., 0., 1.]]), [batch_size, n_points])
    last_row = tf.reshape(last_row, [batch_size, n_points, 1, 4])
    expanded_translations = tf.expand_dims(translations, axis=-1)
    matrices = tf.concat([rot_matrices, expanded_translations], axis=-1)
    matrices = tf.concat(
        [matrices, tf.cast(last_row, matrices.dtype)], axis=-2)
    return matrices


def t_q_to_h_matrix(translations, quaternions):
    rot_matrices = tf_transformation.rotation_matrix_3d.from_quaternion(
        quaternions)
    return t_m_to_h_matrix(translations, rot_matrices)


class LanguageNeRF(tf.keras.Model):
    def __init__(self, n_features=256, original_image_size=(480, 640), n_points_train=5,
                 n_views=1, n_5d_poses=7,
                 pretrained=True, batch_size=1,
                 fixed_orientation=None,
                 rotation_representation='quaternion',
                 softmax_before_loss=False,
                 **kwargs):
        super().__init__(**kwargs)
        if not pretrained:
            raise NotImplementedError(
                'GraspMVNeRF only supports pretrained models')
        self.n_views = n_views
        self.pretrained = pretrained

        self.fine_embedding = MVResNetMLPNeRFEmbedding(6, 128, n_views=n_views,
                                                       embed_direction_vector=True,
                                                       complete_output=True)
        self.visual_features = VisualFeatures(n_features, original_image_size)
        self.clip_visual = CLIPVisualEncoder()
        self.clip_textual = CLIPTextualEncoder()
        self.grasp_readout = GraspReadout(use_bias=True)
        self.combine_clip_visual_features = CombineCLIPVisualV1()

        self.mse = tf.keras.losses.MeanSquaredError()
        self.cosine_similarity = tf.keras.losses.CosineSimilarity(axis=-1)

        self.n_points_train = n_points_train
        self.batch_size = batch_size

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
            Affine(translation=[-base_offset_x, base_offset_y,
                   base_offset_z], rotation=[0.0, np.pi / 2, 0.0]),
            # rf_base
            Affine(translation=[base_offset_x, base_offset_y,
                   base_offset_z], rotation=[0.0, -np.pi / 2, 0.0]),
            # lb_base
            Affine(translation=[-base_offset_x, -base_offset_y,
                   base_offset_z], rotation=[0.0, np.pi / 2, 0.0]),
            # # rb_base
            Affine(translation=[base_offset_x, -base_offset_y,
                   base_offset_z], rotation=[0.0, -np.pi / 2, 0.0])
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

        self.translations = tf.Variable(np.zeros([self.batch_size, self.n_points_train, 3], dtype=np.float32),
                                        trainable=True)
        self.pose_variables = [self.translations]
        self.rotation_representation = rotation_representation
        if self.rotation_representation == 'quaternion':
            self.quaternions = tf.Variable(np.zeros([self.batch_size, self.n_points_train, 4], dtype=np.float32),
                                           trainable=True)
            self.pose_variables.append(self.quaternions)
        elif self.rotation_representation == '6d':
            self.six_d_rotations = tf.Variable(np.zeros([self.batch_size, self.n_points_train, 6], dtype=np.float32),
                                               trainable=True)
            self.pose_variables.append(self.six_d_rotations)
        else:
            raise ValueError(
                'Unknown rotation representation: ' + rotation_representation)

        # self.fixed_orientation = [np.pi, 0.0, np.pi / 2]
        # self.fixed_orientation = fixed_orientation
        # if self.fixed_orientation is None:
        #     self.pose_variables.append(self.quaternions)

        self.landscape_loss_tracker = tf.keras.metrics.Mean(
            name='landscape_loss')
        self.grad_loss_tracker_t = tf.keras.metrics.Mean(name='grad_loss_t')
        self.grad_loss_tracker_r = tf.keras.metrics.Mean(name='grad_loss_r')
        self.pred_tracker = tf.keras.metrics.Mean(name='pred_tracker')

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
        src_images = inputs[4]
        clip_tokens = inputs[7]
        src_images = rearrange(src_images, 'b n h w c -> (b n) h w c')
        clip_images = preprocess_tf(src_images)
        clip_outputs = self.clip_visual(clip_images)
        visual_features = self.visual_features(src_images)
        combined_features = self.combine_clip_visual_features(
            (clip_outputs, visual_features))
        clip_textuals = self.clip_textual(clip_tokens)  # [BN 1024]
        clip_textuals_tiled = tf.expand_dims(
            tf.expand_dims(clip_textuals, axis=1), axis=1)
        clip_textuals_tiled = tf.repeat(clip_textuals_tiled, repeats=tf.shape(
            combined_features)[1], axis=1)  # [BN h w 1024]
        clip_textuals_tiled = tf.repeat(clip_textuals_tiled, repeats=tf.shape(
            combined_features)[2], axis=2)  # [BN h w 1024]
        clip_textuals_tiled = clip_textuals_tiled[..., :256]
        combined_features = tf.math.multiply(
            combined_features, clip_textuals_tiled)  # [BN h w 256]
        combined_features = rearrange(
            combined_features, '(b n) h w c -> b n h w c', n=self.n_views)
        # transforms = t_q_to_h_matrix(self.translations, self.quaternions)
        transforms = self.compute_matrices()
        return self._call(inputs, transforms, self.n_points_train, combined_features)

    def compute_matrices(self):
        if self.rotation_representation == 'quaternion':
            matrices = t_q_to_h_matrix(self.translations, self.quaternions)
        elif self.rotation_representation == '6d':
            r_1 = tf.linalg.normalize(
                self.six_d_rotations[:, :, :3], axis=-1)[0]
            r_2 = tf.linalg.normalize(
                self.six_d_rotations[:, :, 3:], axis=-1)[0]
            r_3 = tf.linalg.cross(r_1, r_2)
            r_est = tf.stack([r_1, r_2, r_3], axis=-1)
            matrices = t_m_to_h_matrix(self.translations, r_est)
        return matrices

    def infer(self, inputs, transforms, n_points_infer, batched_features):
        return self._call(inputs, transforms, n_points_infer, batched_features)

    def _call(self, inputs, transforms, n_points, batched_features):
        # matrices = tf.tile(matrices, [self.batch_size, 1, 1, 1])
        src_images = inputs[4]
        src_intrinsics = inputs[5]
        src_extrinsics_inv = inputs[6]

        normalized_images = src_images * 2 - 1.0
        transforms = transforms[:, tf.newaxis, ...]

        poses = tf.matmul(transforms, self.transforms_to_check)
        translations = poses[..., :3, 3]

        translations_homogeneous = tf.concat(
            [translations, tf.ones_like(translations[..., :1])], axis=-1)
        translations_homogeneous = rearrange(
            translations_homogeneous, 'b n5 np d -> b () d (n5 np)')
        camera_points_homogeneous = tf.matmul(
            src_extrinsics_inv, translations_homogeneous)

        projections = tf.matmul(src_intrinsics, camera_points_homogeneous)
        pixel_locations = tf.math.divide(
            projections[..., :2, :], tf.maximum(projections[..., 2, tf.newaxis, :], 1e-8))
        pixel_locations = tf.clip_by_value(
            pixel_locations, clip_value_min=-1e6, clip_value_max=1e6)

        combined_features = tf.concat(
            [normalized_images, batched_features], axis=-1)
        combined_features = rearrange(
            combined_features, 'b nv w h c -> (b nv) w h c')
        flat_pixel_locations = rearrange(
            pixel_locations, 'b nv d n5np -> (b nv) n5np d')
        combined_features = interpolate_bilinear(
            combined_features, flat_pixel_locations, indexing='xy')

        directions = tf.matmul(poses[..., :3, :3], self.z_dir)
        directions_homogeneous = tf.concat(
            [directions, tf.ones_like(directions[..., :1, :])], axis=-2)
        directions_homogeneous = rearrange(
            directions_homogeneous, 'b n5 np d x -> b () d (n5 np x)')
        camera_directions = tf.matmul(
            src_extrinsics_inv, directions_homogeneous)[..., :3, :]

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

        embeddings = self.fine_embedding(
            [camera_points, camera_directions, combined_features])[4:]
        grasp_success = self.grasp_readout(embeddings)

        return grasp_success

    def set_pose(self, translations, rotations):
        self.translations.assign(translations)
        if self.rotation_representation == 'quaternion':
            self.quaternions.assign(rotations)
        elif self.rotation_representation == '6d':
            self.six_d_rotations.assign(rotations)

    def train_step(self, data):
        inputs, labels = data

        self.set_pose(inputs[0], inputs[1])
        src_images = inputs[4]
        clip_tokens = inputs[7]
        src_images = rearrange(src_images, 'b n h w c -> (b n) h w c')
        visual_features = self.visual_features(src_images)
        clip_images = preprocess_tf(src_images)
        clip_outputs = self.clip_visual(clip_images)
        visual_features = self.visual_features(src_images)
        combined_features = self.combine_clip_visual_features(
            (clip_outputs, visual_features))
        clip_textuals = self.clip_textual(clip_tokens)  # [BN 1024]
        clip_textuals_tiled = tf.expand_dims(
            tf.expand_dims(clip_textuals, axis=1), axis=1)
        clip_textuals_tiled = tf.repeat(clip_textuals_tiled, repeats=tf.shape(
            combined_features)[1], axis=1)  # [BN h w 1024]
        clip_textuals_tiled = tf.repeat(clip_textuals_tiled, repeats=tf.shape(
            combined_features)[2], axis=2)  # [BN h w 1024]
        clip_textuals_tiled = clip_textuals_tiled[..., :256]
        combined_features = tf.math.multiply(
            combined_features, clip_textuals_tiled)  # [BN h w 256]
        combined_features = rearrange(
            combined_features, '(b n) h w c -> b n h w c', n=self.n_views)
        transforms = self.compute_matrices()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.grasp_readout.trainable_variables)
            y_pred = self._call(inputs, transforms,
                                self.n_points_train, combined_features)
            if self.softmax_before_loss:
                y_pred = tf.nn.softmax(y_pred)
            landscape_loss = self.loss(labels[0], y_pred)
            # landscape_gradients = tape.gradient(landscape_loss, self.grasp_readout.trainable_variables)
            # optimize(self.optimizer[0], self.grasp_readout.trainable_variables, landscape_gradients, 1.0)

            self.set_pose(inputs[2], inputs[3])
            # with tf.GradientTape() as tape_2:
            with tf.GradientTape() as tape_1:
                transforms = self.compute_matrices()
                prediction = self._call(
                    inputs, transforms, self.n_points_train, combined_features)
            output_gradients = tape_1.gradient(prediction, self.pose_variables)
            loss_t = self.cosine_similarity(labels[1], output_gradients[0])

            if self.rotation_representation == 'quaternion':
                loss_r = self.cosine_similarity(labels[2], output_gradients[1])
            elif self.rotation_representation == '6d':
                loss_r_1 = self.cosine_similarity(
                    labels[2][..., :3], output_gradients[1][..., :3])
                loss_r_2 = self.cosine_similarity(
                    labels[2][..., 3:], output_gradients[1][..., 3:])
                loss_r = loss_r_1 + loss_r_2

            loss = loss_t + loss_r + landscape_loss
        # gradients = tape_2.gradient(loss, self.grasp_readout.trainable_variables[:-1])
        gradients = tape.gradient(loss, self.grasp_readout.trainable_variables)
        optimize(self.optimizer,
                 self.grasp_readout.trainable_variables, gradients, 1.0)

        self.landscape_loss_tracker.update_state(landscape_loss)
        self.grad_loss_tracker_t.update_state(loss_t)
        self.grad_loss_tracker_r.update_state(loss_r)
        self.pred_tracker.update_state(tf.reduce_mean(prediction))

        return_dict = {
            'landscape_loss': self.landscape_loss_tracker.result(),
            'grad_loss_t': self.grad_loss_tracker_t.result(),
            'grad_loss_r': self.grad_loss_tracker_r.result(),
            'pred': self.pred_tracker.result(),
        }
        return return_dict
