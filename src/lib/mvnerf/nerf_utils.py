import numpy as np
import tensorflow as tf
from einops import rearrange, repeat
import pickle
from tensorflow_addons.image import interpolate_bilinear


def optimize(optimizer, variables, gradients, gradients_clip=0.0):
    if gradients_clip > 0:
        gradients = [tf.clip_by_value(grad, -gradients_clip, gradients_clip) for grad in
                     gradients]
    optimizer.apply_gradients(zip(gradients, variables))


def get_rays(image_width, image_height, extrinsics, intrinsics, norm_direction_vector=True):
    u, v = np.meshgrid(np.arange(image_width, dtype=np.float32),
                       np.arange(image_height, dtype=np.float32), indexing='xy')

    u = rearrange(u, 'h w -> (h w)')
    v = rearrange(v, 'h w -> (h w)')
    rays_o, rays_d = get_specific_rays(u, v, extrinsics, intrinsics, norm_direction_vector)
    rays_d = rearrange(rays_d, '(h w) c -> h w c', h=image_height, w=image_width)
    rays_o = rearrange(rays_o, '(h w) c -> h w c', h=image_height, w=image_width)
    return rays_o, rays_d


def get_specific_rays(u, v, extrinsics, intrinsics, norm_direction_vector=True):
    # directions in camera space
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)
    rays_d = extrinsics[:3, :3] @ np.linalg.inv(intrinsics[:3, :3]) @ pixels
    rays_d = rearrange(rays_d, 'c n -> n c')
    if norm_direction_vector:
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
    rays_o = np.broadcast_to(extrinsics[:3, -1], rays_d.shape)
    return rays_o, rays_d


def bbox_biased_sample(n_sample, bboxes, image_height, image_width, in_box_p=0.8):
    n_inside = int(n_sample * in_box_p)
    n_random = n_sample - n_inside

    in_samples = np.random.randint(bboxes[:2], bboxes[2:], (n_inside, 2))
    random_samples = np.random.randint((image_height, image_width), size=(n_random, 2))

    samples = np.concatenate([in_samples, random_samples], axis=0)
    return samples


def sample_along_ray(rays_origin, rays_direction, near, far, batch_size, n_rays, n_samples):
    step = (far - near) / n_samples
    along_ray_intervals = tf.convert_to_tensor([near + i * step for i in range(n_samples + 1)], dtype=tf.float32)
    lower_bound_values = along_ray_intervals[:-1]

    lower_bound = tf.ones_like(rays_direction)[..., 0]
    lower_bound = repeat(lower_bound, 'b nr -> b nr ns', ns=n_samples)
    lower_bound = lower_bound * lower_bound_values
    random_added = tf.random.uniform(shape=[batch_size, n_rays, n_samples]) * step
    points_along_ray = lower_bound + random_added
    world_points = rays_origin[:, :, tf.newaxis, :] + points_along_ray[..., tf.newaxis] * rays_direction[:, :,
                                                                                          tf.newaxis, :]
    return world_points, points_along_ray


def compute_pixel_in_image_mv(world_points, src_intrinsics, src_extrinsics_inv):
    world_points_homogeneous = tf.concat(
        [world_points, tf.ones((tf.shape(world_points)[0], tf.shape(world_points)[1], tf.shape(world_points)[2], 1))],
        axis=-1)
    world_points_homogeneous_r = rearrange(world_points_homogeneous, 'b nr np wph ->  nr b wph np')
    world_points_homogeneous_r = repeat(world_points_homogeneous_r, 'nr b wph np -> nr nv b wph np', nv=1)
    src_extrinsics_inv_r = rearrange(src_extrinsics_inv, 'b nv w h -> nv b w h')
    camera_points_homogeneous_t_r = tf.matmul(src_extrinsics_inv_r, world_points_homogeneous_r)

    src_intrinsics_r = rearrange(src_intrinsics, 'b nv w h -> nv b w h')
    projections_r = tf.matmul(src_intrinsics_r, camera_points_homogeneous_t_r)
    projections = rearrange(projections_r, 'nr nv b cph np -> b nv nr np cph')
    pixel_locations = tf.math.divide(
        projections[..., :2], tf.maximum(projections[..., 2, tf.newaxis], 1e-8))
    pixel_locations = tf.clip_by_value(pixel_locations, clip_value_min=-1e6, clip_value_max=1e6)

    camera_points_homogeneous = rearrange(camera_points_homogeneous_t_r, 'nr nv b wph np -> b nv nr np wph')
    return pixel_locations, camera_points_homogeneous


def world_to_camera_direction_vector_mv(world_direction_vectors, extrinsics_inverse, n_views):
    """
    Transform a world direction vector to camera direction vector.
    Input
    -----
    - world_direction_vector: tensor of shape (B, 3)
    - extrinsics_inverse: tensor of shape (4, 4)
    Returns
    -------
    - camera_direction_vector: tensor of shape (B, 3)
    """
    world_directions_homogeneous = tf.concat(
        [world_direction_vectors,
         tf.ones((tf.shape(world_direction_vectors)[0], tf.shape(world_direction_vectors)[1], 1))],
        axis=-1)
    world_directions_homogeneous_r = rearrange(world_directions_homogeneous, 'b nr wdh -> b wdh nr')
    # TODO original implementation repeats for n_views. find out why.
    world_directions_homogeneous_r = repeat(world_directions_homogeneous_r, 'b wdh nr -> b nv wdh nr', nv=n_views)
    camera_directions_homogeneous_t_r = tf.matmul(extrinsics_inverse, world_directions_homogeneous_r)
    camera_directions_homogeneous = rearrange(camera_directions_homogeneous_t_r, 'b nv wdh nr -> b nv nr wdh')
    camera_directions = camera_directions_homogeneous[..., :3]
    return camera_directions


def position_encoding(position, n_freq, pos_encoding_freq):
    """
    Position encoding as a function of frequency.
    Input
    -----
    - position: tensor of shape (B, 3)
    - n_freq: number of frequencies
    - pos_encoding_freq: scaling factor for frequencies
    Returns
    -------
    - encoding: tensor of shape (B, 2*n_freq)
    """
    freq_multiplier = pos_encoding_freq * tf.pow(2.0, tf.range(n_freq, dtype='float32'))
    freq_multiplier = rearrange(freq_multiplier, 'n -> () () () () n')
    reshaped_positions = rearrange(position, 'b nr np d -> b nr np d ()')
    encoding = tf.stack([tf.sin(reshaped_positions * freq_multiplier), tf.cos(reshaped_positions * freq_multiplier)],
                        axis=-1)
    encoding = rearrange(encoding, 'b nr np d n f -> b nr np (d n f)')
    return encoding


def sigma_to_alpha(sigma, dists):
    """
    Convert a sigma value to an alpha value.
    Input
    -----
    - sigma: tensor of shape (..., 1)
    Returns
    -------
    - alpha: tensor of shape (..., 1)
    """
    alpha = 1.0 - tf.exp(-dists * tf.keras.activations.relu(sigma))
    return alpha


def sample_pdf(bins, weights, n_samples):
    stable_weights = weights + 1e-5
    w_sum = tf.reduce_sum(stable_weights, axis=-1, keepdims=True)
    w_sum = tf.where(tf.math.abs(w_sum) == 0, tf.ones_like(w_sum), w_sum)
    pdf = stable_weights / w_sum
    cdf = tf.math.cumsum(pdf, axis=-1, exclusive=False)
    cdf = tf.concat([tf.zeros([tf.shape(cdf)[0], tf.shape(cdf)[1], 1]), cdf], axis=-1)

    u = tf.random.uniform((tf.shape(bins)[0], tf.shape(bins)[1], n_samples), 0, 1)
    above_indices = tf.zeros_like(u, dtype='int32')

    cdf_trans = tf.transpose(cdf, perm=[2, 0, 1])

    def greater_than(a, x):
        return a + tf.cast(tf.greater_equal(u, x[..., tf.newaxis]), 'int32')

    scan = tf.scan(greater_than, cdf_trans, initializer=above_indices)
    above_indices = scan[-1]

    below_indices = tf.clip_by_value(above_indices - 1, 0, tf.shape(bins)[-1] - 1)
    cdf = repeat(cdf, 'b nr nb -> b nr n nb', n=n_samples)
    cdf_g_a = tf.gather_nd(cdf, above_indices[..., tf.newaxis], batch_dims=3)
    cdf_g_b = tf.gather_nd(cdf, below_indices[..., tf.newaxis], batch_dims=3)

    bins_ = repeat(bins, 'b nr nb -> b nr n nb', n=n_samples)
    bins_g_a = tf.gather_nd(bins_, above_indices[..., tf.newaxis], batch_dims=3)
    bins_g_b = tf.gather_nd(bins_, below_indices[..., tf.newaxis], batch_dims=3)

    denom = cdf_g_a - cdf_g_b
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g_b) / denom

    samples = bins_g_b + t * (bins_g_a - bins_g_b)
    return samples


def get_transformer_block(weights_dict, block_id, n_heads):
    batch_normalization_gamma = weights_dict[f'blocks.{block_id}.norm1.weight']
    weights_dict.pop(f'blocks.{block_id}.norm1.weight')
    batch_normalization_beta = weights_dict[f'blocks.{block_id}.norm1.bias']
    weights_dict.pop(f'blocks.{block_id}.norm1.bias')
    qkv_weight = weights_dict[f'blocks.{block_id}.attn.qkv.weight']
    weights_dict.pop(f'blocks.{block_id}.attn.qkv.weight')
    qkv_dim = qkv_weight.shape[0] // 3
    embedding_dim = qkv_weight.shape[0] // (3 * n_heads)
    q_weight = qkv_weight[:qkv_dim, :]
    k_weight = qkv_weight[qkv_dim:2 * qkv_dim, :]
    v_weight = qkv_weight[2 * qkv_dim:, :]
    q_weights = q_weight.reshape((n_heads, embedding_dim, -1))
    k_weights = k_weight.reshape((n_heads, embedding_dim, -1))
    v_weights = v_weight.reshape((n_heads, embedding_dim, -1))

    qkv_bias = weights_dict[f'blocks.{block_id}.attn.qkv.bias']
    weights_dict.pop(f'blocks.{block_id}.attn.qkv.bias')
    q_bias = qkv_bias[:qkv_dim]
    k_bias = qkv_bias[qkv_dim:2 * qkv_dim]
    v_bias = qkv_bias[2 * qkv_dim:]
    q_biases = q_bias.reshape((n_heads, -1))
    k_biases = k_bias.reshape((n_heads, -1))
    v_biases = v_bias.reshape((n_heads, -1))

    mha_query_kernel = q_weights.transpose(2, 0, 1)
    mha_query_bias = q_biases
    mha_key_kernel = k_weights.transpose(2, 0, 1)
    mha_key_bias = k_biases
    mha_value_kernel = v_weights.transpose(2, 0, 1)
    mha_value_bias = v_biases

    # TODO check if reshape is not the other way around --> how?
    attention_output_kernel = weights_dict[f'blocks.{block_id}.attn.proj.weight'].reshape(
        (n_heads, embedding_dim, -1))
    weights_dict.pop(f'blocks.{block_id}.attn.proj.weight')
    attention_output_bias = weights_dict[f'blocks.{block_id}.attn.proj.bias']
    weights_dict.pop(f'blocks.{block_id}.attn.proj.bias')

    layer_norm_gamma = weights_dict[f'blocks.{block_id}.norm2.weight']
    weights_dict.pop(f'blocks.{block_id}.norm2.weight')
    layer_norm_beta = weights_dict[f'blocks.{block_id}.norm2.bias']
    weights_dict.pop(f'blocks.{block_id}.norm2.bias')

    dense_0_kernel = weights_dict[f'blocks.{block_id}.mlp.fc1.weight'].transpose(1, 0)
    weights_dict.pop(f'blocks.{block_id}.mlp.fc1.weight')
    dense_0_bias = weights_dict[f'blocks.{block_id}.mlp.fc1.bias']
    weights_dict.pop(f'blocks.{block_id}.mlp.fc1.bias')
    dense_1_kernel = weights_dict[f'blocks.{block_id}.mlp.fc2.weight'].transpose(1, 0)
    weights_dict.pop(f'blocks.{block_id}.mlp.fc2.weight')
    dense_1_bias = weights_dict[f'blocks.{block_id}.mlp.fc2.bias']
    weights_dict.pop(f'blocks.{block_id}.mlp.fc2.bias')

    batch_norm_moving_mean = np.zeros(768)
    batch_norm_moving_variance = np.zeros(768)
    t_block = [
        batch_normalization_gamma,
        batch_normalization_beta,
        mha_query_kernel,
        mha_query_bias,
        mha_key_kernel,
        mha_key_bias,
        mha_value_kernel,
        mha_value_bias,
        attention_output_kernel,
        attention_output_bias,
        layer_norm_gamma,
        layer_norm_beta,
        dense_0_kernel,
        dense_0_bias,
        dense_1_kernel,
        dense_1_bias,
        batch_norm_moving_mean,
        batch_norm_moving_variance
    ]
    return t_block


def load_pretrained_weights(path, vision_transformer):
    with open(path, 'rb') as f:
        torch_weights = pickle.load(f)
    vision_transformer.vit.cls_token.assign(torch_weights['cls_token'])
    torch_weights.pop('cls_token')
    vision_transformer.vit.pos_embedding.assign(torch_weights['pos_embed'])
    torch_weights.pop('pos_embed')

    for i in range(4):
        block_0 = get_transformer_block(torch_weights, i * 3, 12)
        block_1 = get_transformer_block(torch_weights, i * 3 + 1, 12)
        block_2 = get_transformer_block(torch_weights, i * 3 + 2, 12)
        c_block = block_0 + block_1 + block_2
        vision_transformer.vit.transformer_blocks[i].set_weights(c_block)
    vision_transformer.vit.get_layer('patch_embed').proj.set_weights(
        [torch_weights['patch_embed.proj.weight'].transpose((2, 3, 1, 0)), torch_weights['patch_embed.proj.bias']])
    torch_weights.pop('patch_embed.proj.weight')
    torch_weights.pop('patch_embed.proj.bias')


def get_projection_features_mv(inputs, features, pixel_locations, n_rays, n_samples, batch_size):
    combined_features = tf.concat([inputs, features], axis=-1)
    combined_features = rearrange(combined_features, 'b nv w h c -> (b nv) w h c')
    flat_pixel_locations = rearrange(pixel_locations, 'b nv nr np d -> (b nv) (nr np) d')
    # features at pixel locations
    combined_features = interpolate_bilinear(combined_features, flat_pixel_locations, indexing='xy')
    combined_features = rearrange(combined_features, '(b nv) (nr np) c -> b nv nr np c', b=batch_size, nr=n_rays,
                                  np=n_samples)
    return combined_features


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_learning_rate, warmup_steps, scale_down_after=400000):
        self.target_learning_rate = tf.cast(target_learning_rate, tf.float32)
        warmup_steps = tf.cast(warmup_steps, tf.float32)
        # should be minimum 1.0
        self.warmup_steps = tf.math.maximum(1.0, warmup_steps)
        self.scale_down_after = tf.cast(scale_down_after, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.cond(step <= self.warmup_steps, lambda: step / self.warmup_steps * self.target_learning_rate,
                       lambda: tf.cond(step <= self.scale_down_after, lambda: self.target_learning_rate,
                                       lambda: 0.1 * self.target_learning_rate))
