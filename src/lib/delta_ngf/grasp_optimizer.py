import numpy as np
import tensorflow as tf
from einops import rearrange
from manipulation_tasks.transform import Affine
import tensorflow_graphics.geometry.transformation as tf_transformation

from ..mvnerf.nerf_utils import optimize


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


class DNGFOptimizer(tf.keras.Model):
    def __init__(self, nerf_grasper, workspace_bounds, n_initial_guesses=32, n_images=3,
                 fixed_orientation=None, clip_translation=False,
                 rotation_representation='quaternion', **kwargs):
        super(DNGFOptimizer, self).__init__(**kwargs)
        self.workspace_bounds = np.array(workspace_bounds)
        self.nerf_grasper = nerf_grasper
        self.n_initial_guesses = n_initial_guesses
        self.n_images = n_images

        self.batch_size = self.n_images / self.nerf_grasper.n_views
        assert (self.batch_size % 1) == 0
        self.batch_size = int(self.batch_size)

        self.translations = tf.Variable(np.zeros([1, self.n_initial_guesses, 3], dtype=np.float32),
                                        trainable=True)
        self.pose_variables = [[self.translations]]
        self.rotation_representation = rotation_representation
        if self.rotation_representation == 'quaternion':
            self.quaternions = tf.Variable(np.zeros([1, self.n_initial_guesses, 4], dtype=np.float32),
                                           trainable=True)
            self.pose_variables.append([self.quaternions])
        elif self.rotation_representation == '6d':
            self.six_d_rotations = tf.Variable(np.zeros([1, self.n_initial_guesses, 6], dtype=np.float32),
                                               trainable=True)
            self.pose_variables.append([self.six_d_rotations])
        else:
            raise ValueError(
                'Unknown rotation representation: ' + rotation_representation)

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.clip_translation = clip_translation

    def set_initial_guesses(self, initial_guesses):
        assert len(initial_guesses) == 2
        assert initial_guesses[0].shape == (1, self.n_initial_guesses, 3)
        self.translations.assign(initial_guesses[0])
        if self.rotation_representation == 'quaternion':
            assert initial_guesses[1].shape == (1, self.n_initial_guesses, 4)
            self.quaternions.assign(initial_guesses[1])
        elif self.rotation_representation == '6d':
            assert initial_guesses[1].shape == (1, self.n_initial_guesses, 6)
            self.six_d_rotations.assign(initial_guesses[1])

    def generate_initial_guesses(self, workspace_bounds=None, n_initial_guesses=None, batch_size=1):
        if workspace_bounds is None:
            workspace_bounds = self.workspace_bounds
        if n_initial_guesses is None:
            n_initial_guesses = self.n_initial_guesses

        initial_guesses = []
        for b in range(batch_size):
            i_gs = [Affine.random(workspace_bounds,
                                  allow_zero_translation=True,
                                  allow_zero_rotation=True) for _ in range(n_initial_guesses)]
            if len(initial_guesses) == 0:
                initial_guesses = [[], []]
            ts = [pose.translation for pose in i_gs]
            if self.rotation_representation == 'quaternion':
                rs = [pose.quat for pose in i_gs]
            elif self.rotation_representation == '6d':
                rs = [np.concatenate(
                    [pose.rotation[:, 0], pose.rotation[:, 1]]) for pose in i_gs]
            initial_guesses[0].append(ts)
            initial_guesses[1].append(rs)
        initial_guesses = [np.array(i_g) for i_g in initial_guesses]
        return initial_guesses

    def get_results(self):
        matrices = self.compute_matrices()
        transformations = []
        for m in matrices[0]:
            transformations.append(Affine.from_matrix(m.numpy()))

        return transformations

    def call(self, inputs, training=False, mask=None):
        matrices = self.compute_matrices()
        matrices = tf.tile(matrices, [self.batch_size, 1, 1, 1])
        grasp_success = self.nerf_grasper.infer(
            (None, None, None, None, inputs[0][0], inputs[0][1],
             inputs[0][2], inputs[0][3]), matrices, self.n_initial_guesses,
            inputs[1])
        return grasp_success

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

    def post_process(self):
        if self.clip_translation:
            self.translations.assign(tf.clip_by_value(self.translations,
                                                      self.workspace_bounds[:, 0],
                                                      self.workspace_bounds[:, 1]))
        if self.rotation_representation == 'quaternion':
            self.quaternions.assign(
                tf.linalg.normalize(self.quaternions, axis=-1)[0])
        elif self.rotation_representation == '6d':
            r_1 = tf.linalg.normalize(
                self.six_d_rotations[:, :, :3], axis=-1)[0]
            r_2 = tf.linalg.normalize(
                self.six_d_rotations[:, :, 3:], axis=-1)[0]
            self.six_d_rotations.assign(tf.concat([r_1, r_2], axis=-1))

    def compute_current_grasp_success(self, inputs, features):
        if self.batch_size != 1:
            r_inputs = [
                rearrange(inputs[0], 'b nv h w c -> nv b h w c'),
                rearrange(inputs[1], 'b nv r c -> nv b r c'),
                rearrange(inputs[2], 'b nv r c -> nv b r c'),
            ]
            r_features = rearrange(features, 'b nv h w c -> nv b h w c')
        else:
            r_inputs = inputs
            r_features = features

        grasp_success = self([r_inputs, r_features])
        grasp_success = tf.reduce_sum(grasp_success, axis=0, keepdims=True)
        grasp_success = rearrange(grasp_success, 'b np -> np b')
        return grasp_success

    def optimize_pose(self, inputs, features, train_config):
        with tf.GradientTape() as tape:
            # TODO assumes batch size of 1, might need to rework
            if self.batch_size != 1:
                r_inputs = [
                    rearrange(inputs[0], 'b nv h w c -> nv b h w c'),
                    rearrange(inputs[1], 'b nv r c -> nv b r c'),
                    rearrange(inputs[2], 'b nv r c -> nv b r c'),
                    rearrange(inputs[3], 'b nv t -> nv b t'),
                ]
                r_features = rearrange(features, 'b nv h w c -> nv b h w c')
            else:
                r_inputs = inputs
                r_features = features

            grasp_success = self([r_inputs, r_features])
            grasp_success = tf.reduce_sum(grasp_success, axis=0, keepdims=True)
            grasp_success = rearrange(grasp_success, 'b np -> np b')
            loss = -grasp_success

        grads = tape.gradient(loss, self.pose_variables)
        for i, t_c in enumerate(train_config):
            if t_c:
                optimize(self.optimizer[i],
                         self.pose_variables[i], grads[i], 1.0)
        self.loss_tracker.update_state(loss)
        self.post_process()
        return {'loss': grasp_success}
