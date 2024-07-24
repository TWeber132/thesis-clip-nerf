import os

import numpy as np
import tensorflow as tf
from einops import rearrange, repeat

from lib.mvnerf.layers import MVResNetMLPNeRFEmbedding, RenderReadout, VisualFeatures, CombineCLIPVisualV3
from lib.mvnerf.nerf_utils import sample_along_ray, compute_pixel_in_image_mv, get_projection_features_mv, \
    world_to_camera_direction_vector_mv, sigma_to_alpha, sample_pdf, optimize, get_rays
from lib.data_generator.mvnerf import MVNeRFDataGenerator
from lib.clip.model import CLIPVisualEncoder
from lib.clip.utils import preprocess_tf


class MVVNeRFRenderer(tf.keras.Model):
    def __init__(self, n_rays_train, n_rays_infer, n_views=2, n_samples=64, n_features=256,
                 embed_direction_vector=True,
                 batch_size=1, near=0.7, far=1.5, original_image_size=(480, 640), **kwargs):
        super(MVVNeRFRenderer, self).__init__(**kwargs)
        self.n_views = n_views

        self.coarse_embedding = MVResNetMLPNeRFEmbedding(6, 128, embed_direction_vector=embed_direction_vector,
                                                         n_views=n_views)
        self.coarse_readout = RenderReadout(4)
        self.fine_embedding = MVResNetMLPNeRFEmbedding(6, 128, embed_direction_vector=embed_direction_vector,
                                                       n_views=n_views)
        self.fine_readout = RenderReadout(4)

        self.visual_features = VisualFeatures(n_features, original_image_size)
        self.clip_visual = CLIPVisualEncoder()
        self.combine_clip_visual_features = CombineCLIPVisualV3()
        self.up_sample = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

        self.n_samples = n_samples
        self.n_rays_train = n_rays_train
        self.n_rays_infer = n_rays_infer

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        self.batch_size = batch_size
        self.infer_batch_size = 1

        self.near = near
        self.far = far

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32, name="images")])
    def encode(self, image):
        features = self.visual_features(image)
        return features

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, 512, 3), dtype=tf.float32, name="ray_origins"),
                                   tf.TensorSpec(
                                       shape=(None, 512, 3), dtype=tf.float32, name="ray_directions"),
                                   tf.TensorSpec(
                                       shape=(None, None, 480, 640, 3), dtype=tf.float32, name="images"),
                                   tf.TensorSpec(
                                       shape=(None, None, 4, 4), dtype=tf.float32, name="intrinsics"),
                                   tf.TensorSpec(
                                       shape=(None, None, 4, 4), dtype=tf.float32, name="extrinsics_inv")),
                                  tf.TensorSpec(shape=(None, None, 480, 640, 256), dtype=tf.float32, name="combined_features")])
    def infer(self, inputs, batched_features):
        # TODO n_rays_infer should come from outside ...
        return self._call(inputs, self.n_rays_infer, self.infer_batch_size, batched_features)

    @ tf.function(input_signature=[(tf.TensorSpec(shape=(None, 512, 3), dtype=tf.float32, name="ray_origins"),
                                   tf.TensorSpec(
                                       shape=(None, 512, 3), dtype=tf.float32, name="ray_directions"),
                                   tf.TensorSpec(
                                       shape=(None, None, 480, 640, 3), dtype=tf.float32, name="images"),
                                   tf.TensorSpec(
                                       shape=(None, None, 4, 4), dtype=tf.float32, name="intrinsics"),
                                   tf.TensorSpec(
                                       shape=(None, None, 4, 4), dtype=tf.float32, name="extrinsics_inv")),
                                   tf.TensorSpec(shape=(), dtype=tf.bool, name="training")])
    def call(self, inputs, training=False, mask=None):
        src_images = inputs[2]
        src_images = rearrange(src_images, 'b n h w c -> (b n) h w c')

        # Combine CLIP and VISUAL
        clip_images = preprocess_tf(src_images)
        clip_outputs = self.clip_visual(clip_images)
        visual_features = self.encode(src_images)
        outputs = self.combine_clip_visual_features(  # [(BN) 240 320 256] [(BN) 120 160 256] [(BN) 60 80 512] [(BN) 30 40 1024]
            (clip_outputs, visual_features))
        combined_features = self.up_sample(outputs[0])  # [(BN) 480 640 256]
        combined_features = rearrange(
            combined_features, '(b n) h w c -> b n h w c', b=self.batch_size)
        return self._call(inputs, self.n_rays_train, self.batch_size, combined_features)

    @ staticmethod
    def volumetric_render(zs, density, chromacity):
        dists = zs[..., 1:] - zs[..., :-1]
        dists = tf.concat([dists, dists[..., -1:]], axis=-1)
        alpha = sigma_to_alpha(density, dists)
        transmittance = tf.math.cumprod(
            1 - alpha + 1e-10, axis=-1, exclusive=True)
        weights = alpha * transmittance
        rgb = tf.reduce_sum(
            rearrange(weights, 'b nr np -> b nr np ()') * chromacity, axis=-2)
        depth = tf.reduce_sum(weights * zs, axis=-1)
        return rgb, depth, weights

    @ tf.function(input_signature=[(tf.TensorSpec(shape=(None, 512, 3), dtype=tf.float32, name="ray_origins"),
                                   tf.TensorSpec(
                                       shape=(None, 512, 3), dtype=tf.float32, name="ray_directions"),
                                   tf.TensorSpec(
                                       shape=(None, None, 480, 640, 3), dtype=tf.float32, name="images"),
                                   tf.TensorSpec(
                                       shape=(None, None, 4, 4), dtype=tf.float32, name="intrinsics"),
                                   tf.TensorSpec(shape=(None, None, 4, 4), dtype=tf.float32, name="extrinsics_inv")),
                                   tf.TensorSpec(shape=(), dtype=tf.int32, name="n_rays"), tf.TensorSpec(
                                       shape=(), dtype=tf.int32, name="batch_size"),
                                   tf.TensorSpec(shape=(None, None, 480, 640, 256), dtype=tf.float32, name="combined_features")])
    def _call(self, inputs, n_rays, batch_size, combined_features):
        ray_origins = inputs[0]
        ray_directions = inputs[1]
        src_images = inputs[2]
        src_intrinsics = inputs[3]
        src_extrinsics_inv = inputs[4]

        normalized_images = src_images * 2 - 1.0
        # get points along rays in world coordinates and as a distance (from near to far) along the ray
        world_points, points_along_ray = sample_along_ray(ray_origins, ray_directions,
                                                          self.near, self.far, batch_size, n_rays, self.n_samples)
        # compute the pixel in the image that corresponds to each point along the ray as well as their representation
        # in source camera coordinates
        pixel_locations, camera_points_homogeneous = compute_pixel_in_image_mv(world_points,
                                                                               src_intrinsics,
                                                                               src_extrinsics_inv)
        features = get_projection_features_mv(normalized_images, combined_features, pixel_locations, n_rays,
                                              self.n_samples, batch_size)
        # transform rays direction from world to src camera coordinates
        camera_directions = world_to_camera_direction_vector_mv(
            ray_directions, src_extrinsics_inv, self.n_views)
        repeated_camera_directions = repeat(
            camera_directions, 'b nv nr d -> b nv nr np d', np=self.n_samples)
        repeated_camera_directions = rearrange(
            repeated_camera_directions, 'b nv nr np d -> (b nv) nr np d')
        camera_points_homogeneous = rearrange(
            camera_points_homogeneous, 'b nv nr np d -> (b nv) nr np d')
        features = rearrange(features, 'b nv nr np d -> (b nv) nr np d')
        # compute coarse chromacity and density
        coarse_embedding = self.coarse_embedding(
            (camera_points_homogeneous[..., :3], repeated_camera_directions, features))
        coarse_chromacity, coarse_density = self.coarse_readout(
            coarse_embedding)
        # render rays
        rgb, depth, weights = MVVNeRFRenderer.volumetric_render(points_along_ray, coarse_density,
                                                                coarse_chromacity)
        # resample points based on estimated density
        z_vals_mid = 0.5 * \
            (points_along_ray[..., 1:] + points_along_ray[..., :-1])
        probs = weights[..., 1:-1]
        z_samples = sample_pdf(z_vals_mid, probs, self.n_samples)
        # compute fine sampled points
        all_zs = tf.concat([points_along_ray, z_samples], axis=-1)
        all_zs = tf.sort(all_zs, axis=-1)
        fine_points = ray_origins[:, :, tf.newaxis, :] + \
            all_zs[..., tf.newaxis] * ray_directions[:, :, tf.newaxis, :]

        # repeat the whole process for the fine points (twice as many samples)
        fine_pixel_locations, fine_camera_points_homogeneous = compute_pixel_in_image_mv(fine_points,
                                                                                         src_intrinsics,
                                                                                         src_extrinsics_inv)
        fine_features = get_projection_features_mv(normalized_images, combined_features, fine_pixel_locations, n_rays,
                                                   self.n_samples * 2, batch_size)

        fine_repeated_camera_directions = repeat(
            camera_directions, 'b nv nr d -> b nv nr np d', np=self.n_samples * 2)
        fine_repeated_camera_directions = rearrange(
            fine_repeated_camera_directions, 'b nv nr np d -> (b nv) nr np d')
        fine_camera_points_homogeneous = rearrange(
            fine_camera_points_homogeneous, 'b nv nr np d -> (b nv) nr np d')
        fine_features = rearrange(
            fine_features, 'b nv nr np d -> (b nv) nr np d')

        fine_camera_points = fine_camera_points_homogeneous[..., :3]

        fine_embedding = self.fine_embedding(
            (fine_camera_points, fine_repeated_camera_directions, fine_features))
        fine_chromacity, fine_density = self.fine_readout(fine_embedding)

        fine_rgb, fine_depth, _ = MVVNeRFRenderer.volumetric_render(
            all_zs, fine_density, fine_chromacity)
        return rgb, depth, fine_rgb, fine_depth

    @tf.function()
    def train_step(self, data):
        inputs, labels = data

        with tf.GradientTape() as tape:
            rgb, depth, fine_rgb, fine_depth = self.call(
                inputs, training=True)
            loss = self.loss(labels, rgb) + self.loss(labels, fine_rgb)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimize(self.optimizer, self.trainable_variables, gradients, 1.0)
        self.loss_tracker.update_state(loss)
        return {'loss': loss}

    def store(self, path):
        coarse_embedding_path = f'{path}_coarse_embedding'
        self.coarse_embedding.save_weights(coarse_embedding_path)
        coarse_readout_path = f'{path}_coarse_readout'
        self.coarse_readout.save_weights(coarse_readout_path)
        fine_embedding_path = f'{path}_fine_embedding'
        self.fine_embedding.save_weights(fine_embedding_path)
        fine_readout_path = f'{path}_fine_readout'
        self.fine_readout.save_weights(fine_readout_path)
        visual_features_path = f'{path}_visual_features'
        self.visual_features.save_weights(visual_features_path)

    def load(self, path, old=False):
        coarse_embedding_path = f'{path}_coarse_embedding'
        coarse_readout_path = f'{path}_coarse_readout'
        fine_embedding_path = f'{path}_fine_embedding'
        fine_readout_path = f'{path}_fine_readout'
        visual_features_path = f'{path}_visual_features'
        # check if <path>.index exists for all paths above
        if not os.path.exists(coarse_embedding_path + '.index'):
            return False
        if not os.path.exists(coarse_readout_path + '.index'):
            return False
        if not os.path.exists(fine_embedding_path + '.index'):
            return False
        if not os.path.exists(fine_readout_path + '.index'):
            return False
        if not os.path.exists(visual_features_path + '.index'):
            return False

        self.coarse_embedding.load_weights(coarse_embedding_path)
        self.coarse_readout.load_weights(coarse_readout_path)
        self.fine_embedding.load_weights(fine_embedding_path)
        self.fine_readout.load_weights(fine_readout_path)
        self.visual_features.load_weights(visual_features_path)
        return True


def render_view(model, src_colors, src_camera_configs, tgt_camera_config):
    tgt_intrinsic = np.reshape(tgt_camera_config['intrinsics'], (3, 3))
    tgt_intrinsic = tgt_intrinsic.astype(np.float32)

    image_shape = src_colors[0].shape[:2]
    tgt_r_o, tgt_r_d = get_rays(
        image_shape[1], image_shape[0], tgt_camera_config['pose'], tgt_intrinsic)
    all_ray_os = np.reshape(tgt_r_o, (-1, 3))
    all_ray_ds = np.reshape(tgt_r_d, (-1, 3))
    all_rgbs = []
    all_depths = []

    src_images = np.array([[image[..., :3] / 255.0 for image in src_colors]])

    src_images = rearrange(src_images, 'b n h w c -> (b n) h w c')
    visual_features = model.encode(src_images)
    clip_images = preprocess_tf(src_images)
    clip_outputs = model.clip_visual(clip_images)
    outputs = model.combine_clip_visual_features((
        clip_outputs, visual_features))
    combined_features = model.up_sample(outputs[0])  # [(BN) 480 640 256]
    combined_features = rearrange(
        combined_features, '(b n) h w c -> b n h w c', b=1)

    for i in range(0, len(all_ray_os), model.n_rays_infer):
        rays_o = all_ray_os[i:i + model.n_rays_infer]
        rays_d = all_ray_ds[i:i + model.n_rays_infer]
        nn_input = MVNeRFDataGenerator.get_input(
            src_images, src_camera_configs, rays_d, rays_o)
        rgb, depth, fine_rgb, fine_depth = model.infer(
            nn_input, combined_features)
        all_rgbs.append(fine_rgb.numpy())
        all_depths.append(fine_depth.numpy())
    all_rgbs = np.reshape(np.array(all_rgbs), (*image_shape, 3)) * 255
    all_rgbs = np.clip(all_rgbs, 0, 255).astype(np.uint8)
    all_depths = np.reshape(np.array(all_depths), (*image_shape, 1))
    norm_depths = (all_depths - np.min(all_depths)) / \
        (np.max(all_depths) - np.min(all_depths))
    all_depths = (norm_depths * 255).astype(np.uint8)
    return all_rgbs, all_depths
