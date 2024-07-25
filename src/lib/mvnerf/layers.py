import tensorflow as tf
import numpy as np
from lib.mvnerf.nerf_utils import position_encoding
from einops import rearrange


class Block(tf.keras.layers.Layer):
    def __init__(self, n_features, downsample=None, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(n_features, 3, padding='same')
        self.norm_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(n_features, 3, padding='same')
        self.norm_1 = tf.keras.layers.BatchNormalization()

        self.downsample = downsample
        # TODO apply prelu here, too. move downsample to sequential in conv encoder
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=False, mask=None):
        skip = inputs
        out = self.conv_1(inputs)
        out = self.norm_1(out, training=True)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.norm_1(out, training=True)

        if self.downsample is not None:
            skip = self.downsample(inputs)

        out += skip
        out = self.relu(out)
        return out


class ConvolutionalEncoder(tf.keras.Model):
    def __init__(self, n_features, **kwargs):
        super(ConvolutionalEncoder, self).__init__(**kwargs)

        downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                n_features // 2, kernel_size=1, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.conv_features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=7, padding='same', use_bias=False, strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            Block(n_features // 2, downsample),
            Block(n_features // 2, None),
            Block(n_features // 2, None),
        ])

    def call(self, inputs, training=False, mask=None):
        return self.conv_features(inputs, training=training)


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=(patch_size, patch_size),
                                           strides=(patch_size, patch_size))

    def call(self, inputs, **kwargs):
        x = self.proj(inputs)
        # x = rearrange(x, 'b h w c -> b (h w) c')
        return x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=12, embed_dim=768, mlp_ratio=4, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.layer_norm_1 = tf.keras.layers.BatchNormalization()
        # in order to make analogy with th pytorch attention implementation
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads,
                                                            value_dim=embed_dim // num_heads)

        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(
                embed_dim * mlp_ratio, activation='gelu', name=f'{self.name}/dense_0'),
            tf.keras.layers.Dense(embed_dim, name=f'{self.name}/dense_1')
        ])

    def call(self, inputs, training=False, mask=None):
        x = self.layer_norm_1(inputs)
        x = self.attention(x, x)
        x = inputs + x
        x = self.layer_norm_2(x)
        x = self.mlp(x)
        x = inputs + x
        return x


class VisionTransformer(tf.keras.Model):
    def __init__(self,
                 img_size=(224, 224),
                 patch_size=16,
                 embed_dim=768,
                 mlp_ratio=4,
                 num_classes=1000,
                 hooks=(3, 6, 9, 12),
                 skip_classification=True,
                 num_heads=12,
                 **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.image_size = img_size
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]
        # here might be a problem with batch size
        self.cls_token = tf.Variable(
            tf.zeros([1, 1, embed_dim]), trainable=True, name='cls_token')
        embed_len = num_patches + 1
        self.pos_embedding = tf.Variable(tf.random.normal([1, embed_len, embed_dim]) * 0.02, trainable=True,
                                         name='pos_embedding')

        # TODO not really general ... need refactoring
        feature_hooks = [hooks[0]] + [hooks[i] - hooks[i - 1]
                                      for i in range(1, len(hooks))]
        self.transformer_blocks = [
            tf.keras.Sequential(
                [TransformerBlock(num_heads, embed_dim, mlp_ratio, name=f't_block_{j * h + i}') for i in range(h)]) for
            j, h in enumerate(feature_hooks)]

        self.skip_classification = skip_classification

        if not self.skip_classification:
            self.norm = tf.keras.layers.LayerNormalization()
            self.head = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False, mask=None):
        x = self.patch_embed(inputs)
        x = rearrange(x, 'b h w c -> b (h w) c')
        cls_token = tf.broadcast_to(self.cls_token,
                                    [tf.shape(x)[0], tf.shape(self.cls_token)[1], tf.shape(self.cls_token)[2]])
        x = tf.concat([cls_token, x], axis=1)
        pos_embedding = tf.broadcast_to(self.pos_embedding,
                                        [tf.shape(x)[0], tf.shape(self.pos_embedding)[1],
                                         tf.shape(self.pos_embedding)[2]])
        x = x + pos_embedding
        features = [self.transformer_blocks[0](x)]
        for block in self.transformer_blocks[1:]:
            features.append(block(features[-1]))

        if not self.skip_classification:
            x = self.norm(x)
            x = self.head(x[:, 0])
        return x, features


class VisionTransformerEncoder(tf.keras.Model):
    def __init__(self,
                 img_size=(224, 224),
                 patch_size=16,
                 embed_dim=768,
                 n_features=256,
                 mlp_ratio=4,
                 num_classes=1000,
                 hooks=(3, 6, 9, 12),
                 features=(48, 96, 192, 384),
                 **kwargs):
        super(VisionTransformerEncoder, self).__init__(**kwargs)

        self.vit = VisionTransformer(
            img_size, patch_size, embed_dim, mlp_ratio, num_classes, hooks)
        self.post_process_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(features[0], kernel_size=1),
            tf.keras.layers.Conv2DTranspose(
                features[0], kernel_size=4, strides=4)
        ])
        self.post_process_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(features[1], kernel_size=1),
            tf.keras.layers.Conv2DTranspose(
                features[1], kernel_size=2, strides=2)
        ])
        self.post_process_3 = tf.keras.layers.Conv2D(
            features[2], kernel_size=1)
        self.post_process_4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(features[3], kernel_size=1),
            tf.keras.layers.Conv2D(
                features[3], kernel_size=3, strides=2, padding='same')
        ])

        self.conv_decode_1 = tf.keras.layers.Conv2D(
            n_features, kernel_size=3, padding='same', use_bias=False)
        self.conv_decode_2 = tf.keras.layers.Conv2D(
            n_features, kernel_size=3, padding='same', use_bias=False)
        self.conv_decode_3 = tf.keras.layers.Conv2D(
            n_features, kernel_size=3, padding='same', use_bias=False)
        self.conv_decode_4 = tf.keras.layers.Conv2D(
            n_features, kernel_size=3, padding='same', use_bias=False)

        self.upsample_1 = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')
        self.upsample_2 = tf.keras.layers.UpSampling2D(
            size=(4, 4), interpolation='bilinear')
        self.upsample_3 = tf.keras.layers.UpSampling2D(
            size=(8, 8), interpolation='bilinear')
        self.upsample_4 = tf.keras.layers.UpSampling2D(
            size=(16, 16), interpolation='bilinear')

        self.output_conv = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_features, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                n_features // 2, kernel_size=3, padding='same')
        ])

    def call(self, inputs, training=False, mask=None):
        _, features = self.vit(inputs)
        features = [rearrange(f[:, 1:], 'b (h w) c -> b h w c', h=self.vit.grid_size[0], w=self.vit.grid_size[1]) for f
                    in features]
        latents = tf.concat([
            self.upsample_1(self.conv_decode_1(
                self.post_process_1(features[0]))),
            self.upsample_2(self.conv_decode_2(
                self.post_process_2(features[1]))),
            self.upsample_3(self.conv_decode_3(
                self.post_process_3(features[2]))),
            self.upsample_4(self.conv_decode_4(
                self.post_process_4(features[3])))
        ], axis=-1)
        latents = self.output_conv(latents)
        return latents


class VisualFeatures(tf.keras.Model):
    def __init__(self, n_features=256, original_image_size=(480, 640), **kwargs):
        super(VisualFeatures, self).__init__(**kwargs)

        self.conv_features = ConvolutionalEncoder(n_features=n_features)

        self.vision_transformer = VisionTransformerEncoder(
            n_features=n_features)

        transformer_image_size = (224, 224)
        self.transformer_downscale = tf.keras.layers.Resizing(transformer_image_size[0], transformer_image_size[1],
                                                              interpolation='bilinear')
        self.transformer_upscale = tf.keras.layers.Resizing(original_image_size[0] // 2, original_image_size[1] // 2,
                                                            interpolation='bilinear')

        # self.feature_upsample = tf.keras.layers.UpSampling2D(
        #     size=(2, 2), interpolation='bilinear')

    def call(self, inputs, training=False, mask=None):
        # b h w c
        image_transformer = self.transformer_downscale(inputs)
        latents = self.vision_transformer(image_transformer)
        # scale image back to half the original size
        latents = self.transformer_upscale(latents)
        skip_latents = self.conv_features(inputs)
        features = tf.concat([latents, skip_latents], axis=-1)
        # features = self.feature_upsample(features)
        return features


class ResNetMLPBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, transform_shortcut=False, activation='relu',
                 kernel_initializer='glorot_uniform', **kwargs):
        super(ResNetMLPBlock, self).__init__(**kwargs)

        self.layer_0 = tf.keras.layers.Dense(
            hidden_size, kernel_initializer=kernel_initializer)
        self.layer_1 = tf.keras.layers.Dense(
            output_size, kernel_initializer=kernel_initializer)

        if activation == 'relu':
            self.prelu_0 = tf.keras.layers.ReLU()
            self.prelu_1 = tf.keras.layers.ReLU()
        elif activation == 'elu':
            self.prelu_0 = tf.keras.layers.ELU()
            self.prelu_1 = tf.keras.layers.ELU()
        else:
            raise ValueError(f'activation {activation} not supported')

        if transform_shortcut:
            self.shortcut = tf.keras.layers.Dense(
                output_size, use_bias=False, kernel_initializer=kernel_initializer)
        else:
            self.shortcut = None

    def call(self, inputs, training=False, mask=None):
        residual = self.prelu_0(inputs)
        residual = self.layer_0(residual)
        residual = self.prelu_1(residual)
        residual = self.layer_1(residual)

        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
        output = shortcut + residual
        return output


class ResNetMLPNeRFEmbedding(tf.keras.Model):
    def __init__(self, n_blocks, hidden_layer_size, n_freq=10, pos_encoding_freq=np.pi,
                 embed_direction_vector=False, complete_output=False, **kwargs):
        super(ResNetMLPNeRFEmbedding, self).__init__(**kwargs)
        self.n_freq = n_freq
        self.pos_encoding_freq = pos_encoding_freq
        self.embed_direction_vector = embed_direction_vector

        self.layer_0 = tf.keras.layers.Dense(hidden_layer_size)
        self.resnet_blocks = [ResNetMLPBlock(hidden_layer_size, hidden_layer_size) for _ in
                              range(n_blocks)]

        self.complete_output = complete_output

    def call(self, inputs, training=False, mask=None):
        encoded_pos = position_encoding(
            inputs[0], self.n_freq, self.pos_encoding_freq)
        if self.embed_direction_vector:
            encoded_dir = position_encoding(
                inputs[1], self.n_freq, self.pos_encoding_freq)
        else:
            encoded_dir = inputs[1]
        x = tf.concat([encoded_pos, encoded_dir, inputs[2]], axis=-1)
        x = self.layer_0(x)
        outputs = [x]
        for block in self.resnet_blocks:
            outputs.append(block(outputs[-1]))
        if self.complete_output:
            return outputs
        else:
            return outputs[-1]


class MVResNetMLPNeRFEmbedding(tf.keras.Model):
    def __init__(self, n_blocks, hidden_layer_size, n_views=2, n_freq=10, pos_encoding_freq=np.pi,
                 embed_direction_vector=False, complete_output=False, **kwargs):
        super(MVResNetMLPNeRFEmbedding, self).__init__(**kwargs)
        self.n_freq = n_freq
        self.pos_encoding_freq = pos_encoding_freq
        self.embed_direction_vector = embed_direction_vector
        self.n_views = n_views

        n_features_blocks = n_blocks // 2
        n_fusion_blocks = n_blocks - n_features_blocks

        self.layer_0 = tf.keras.layers.Dense(hidden_layer_size)
        self.feature_blocks = [ResNetMLPBlock(hidden_layer_size, hidden_layer_size) for _ in
                               range(n_features_blocks)]
        self.fusion_blocks = [ResNetMLPBlock(hidden_layer_size, hidden_layer_size) for _ in
                              range(n_fusion_blocks)]

        self.complete_output = complete_output

    def call(self, inputs, training=False, mask=None):
        encoded_pos = position_encoding(
            inputs[0], self.n_freq, self.pos_encoding_freq)
        if self.embed_direction_vector:
            encoded_dir = position_encoding(
                inputs[1], self.n_freq, self.pos_encoding_freq)
        else:
            encoded_dir = inputs[1]
        x = tf.concat([encoded_pos, encoded_dir, inputs[2]], axis=-1)
        x = self.layer_0(x)
        outputs = [x]
        for block in self.feature_blocks:
            outputs.append(block(outputs[-1]))

        pre_fusion = rearrange(
            outputs[-1], '(b nv) nr np d -> b nv nr np d', nv=self.n_views)
        fusion = tf.reduce_mean(pre_fusion, axis=1)
        outputs.append(fusion)

        for block in self.fusion_blocks:
            outputs.append(block(outputs[-1]))

        if self.complete_output:
            return outputs
        else:
            return outputs[-1]


class RenderReadout(tf.keras.Model):
    def __init__(self, output_size, **kwargs):
        super(RenderReadout, self).__init__(**kwargs)
        self.prelu = tf.keras.layers.ReLU()
        self.output_layer = tf.keras.layers.Dense(output_size)

        self.softplus = tf.keras.layers.Activation(
            tf.keras.activations.softplus)
        self.sigmoid = tf.keras.layers.Activation(tf.keras.activations.sigmoid)

    def call(self, inputs, training=False, mask=None):
        output = self.prelu(inputs)
        output = self.output_layer(output)
        chromacity = self.sigmoid(output[..., :3])
        density = self.softplus(output[..., 3])
        return chromacity, density


class Readout(tf.keras.Model):
    def __init__(self, output_size, use_bias=True,
                 kernel_initializer='glorot_uniform', **kwargs):
        super(Readout, self).__init__(**kwargs)
        self.prelu = tf.keras.layers.ReLU()
        self.output_layer = tf.keras.layers.Dense(
            output_size, use_bias=use_bias, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False, mask=None):
        output = self.prelu(inputs)
        output = self.output_layer(output)
        return output


class Level(tf.keras.layers.Layer):
    def __init__(self, downscale=1, vis_size=(240, 320), filters=256, name="level"):
        super().__init__(name=name)
        self.vis_size = vis_size
        self.filters = filters

        self.resize_down = tf.keras.layers.Resizing(
            vis_size[0] // downscale,
            vis_size[1] // downscale,
            interpolation='bilinear')
        self.resize_up = tf.keras.layers.Resizing(
            vis_size[0],
            vis_size[1],
            interpolation='bilinear')
        self.pre_conv = tf.keras.layers.Conv2D(filters=filters,
                                               kernel_size=1,
                                               use_bias=False)
        self.post_conv = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=1,
                                                use_bias=False)

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        clip_x = inputs[0]
        vis = inputs[1]

        clip_x = self.resize_down(self.pre_conv(clip_x))
        vis = self.resize_down(vis)
        x = tf.concat([clip_x, vis], axis=-1)
        x = self.resize_up(self.post_conv(x))
        return x


class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, filters, name="double_conv"):
        super().__init__(name=name)
        self.conv_1 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding='same', use_bias=False)
        self.relu_1 = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding='same', use_bias=False)
        self.relu_2 = tf.keras.layers.ReLU()

    @tf.function(reduce_retracing=True)
    def call(self, x):
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        return x


class Up(tf.keras.layers.Layer):
    def __init__(self, shape, name="level"):
        super().__init__(name=name)
        self.shape = shape

        self.resize = tf.keras.layers.Resizing(
            self.shape[0], self.shape[1], interpolation='bilinear')
        self.upsample = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')
        self.double_conv = DoubleConv(filters=self.shape[2])
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=self.shape[2], kernel_size=1, padding='same', use_bias=False)

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        clip_bx = inputs[0]
        clip_x = inputs[1]

        clip_bx = self.upsample(clip_bx)
        clip_x = self.resize(clip_x)
        clip = tf.concat([clip_bx, clip_x], axis=-1)
        clip = self.double_conv(clip)
        return clip


class CLIPFeatureExtraction(tf.keras.layers.Layer):
    def __init__(self, shape=(240, 320, 256), name="clip_feature_extraction"):
        super().__init__(name=name)
        assert shape[2] == 256, f"Expected 256 input channels but got {shape[3]} which does not lead to 1024 output channels."
        pool_size = (shape[0] // 2, shape[1] // 2)

        # 4 features per feature map
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_size, padding='valid')
        self.flatten = tf.keras.layers.Flatten()

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        return self.flatten(self.max_pool(inputs))


class ConvFusion(tf.keras.layers.Layer):
    def __init__(self, filters, name="conv_fusion"):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, use_bias=False)
        self.relu = tf.keras.layers.ReLU()

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        x1, x2 = inputs
        x = tf.concat([x1, x2], axis=-1)    # [B, H, W, 2C]
        x = self.relu(x)
        x = self.conv(x)                    # [B, H, W, C]
        return x


class MultiplyFusion(tf.keras.layers.Layer):
    def __init__(self, shape=(240, 320), filters=256, use_dense=True, name="multiply_fusion"):
        super().__init__(name=name)

        self.tile = Tile(shape=shape, filters=filters, use_dense=use_dense)

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        clip_x = inputs[0]
        clip_textuals = inputs[1]
        clip_t = self.tile(clip_textuals)
        return tf.math.multiply(clip_x, clip_t)


class Tile(tf.keras.layers.Layer):
    def __init__(self, shape=(240, 320), filters=256, use_dense=True, name="tile", ):
        super().__init__(name=name)

        self.shape = shape
        self.dense = tf.keras.layers.Dense(
            units=filters, use_bias=False) if use_dense else Slice(filters)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_textuals")])
    def call(self, inputs):
        x = self.dense(inputs)
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # [(BN) 1 1 F]
        x = tf.repeat(x, repeats=self.shape[0], axis=1)  # [(BN) H 1 F]
        x = tf.repeat(x, repeats=self.shape[1], axis=2)  # [(BN) H W F]
        return x


class Slice(tf.keras.layers.Layer):
    def __init__(self, len=256, name="slice"):
        super().__init__(name=name)

        self.begin = [0, 0]
        self.end = [-1, len]

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.end)


class CombineCLIPVisualV3(tf.keras.Model):
    def __init__(self, use_dense=False, name="combine_clip_visual"):
        super().__init__(name=name)

        self.resize_1 = tf.keras.layers.Resizing(
            120, 160, interpolation='bilinear')
        self.resize_2 = tf.keras.layers.Resizing(
            60, 80, interpolation='bilinear')
        self.resize_3 = tf.keras.layers.Resizing(
            30, 40, interpolation='bilinear')
        self.conv = tf.keras.layers.Conv2D(
            1024, kernel_size=3, padding='same', use_bias=False, activation='relu')
        self.multiply_fusion_1 = MultiplyFusion(
            (30, 40), filters=1024, use_dense=use_dense)
        self.up_1 = Up(shape=(60, 80, 512))
        self.multiply_fusion_2 = MultiplyFusion(
            (60, 80), filters=512, use_dense=use_dense)
        self.conv_fusion_1 = ConvFusion(filters=512)
        self.up_2 = Up(shape=(120, 160, 256))
        self.multiply_fusion_3 = MultiplyFusion(
            (120, 160), filters=256, use_dense=use_dense)
        self.conv_fusion_2 = ConvFusion(filters=256)
        self.up_3 = Up(shape=(240, 320, 256))
        self.conv_fusion_3 = ConvFusion(filters=256)
        self.up_sample = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

    @tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_visuals"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(
                                       shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"),
                                   tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_textuals"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        visual_features = inputs[1]                 # [(BN) 240 320 256]
        clip_textuals = inputs[2]                   # [(BN) 1024]
        vis_1 = self.resize_1(visual_features)      # [(BN) 120 160 256]
        vis_2 = self.resize_2(visual_features)      # [(BN) 60 80 256]
        _clip_visuals = clip_outputs[0]             # [(BN) 1024]
        clip_l1 = clip_outputs[1]                   # [(BN) 56 56 256]
        clip_l2 = clip_outputs[2]                   # [(BN) 28 28 512]
        clip_l3 = clip_outputs[3]                   # [(BN) 14 14 1024]
        clip_l4 = clip_outputs[4]                   # [(BN) 7 7 2048]

        x = self.conv(self.resize_3(clip_l4))       # [(BN) 30, 40, 1024]
        x = self.multiply_fusion_1((x, clip_textuals))  # [(BN) 30, 40, 1024]
        # No fusion: clip | vis => 1 | -
        x = self.up_1((x, clip_l3))                 # [(BN) 60 80 512]
        x = self.multiply_fusion_2((x, clip_textuals))  # [(BN) 60 80 512]
        # Fusion: clip | vis => 2 | 1
        x = self.conv_fusion_1((x, vis_2))          # [(BN) 60 80 512]
        x = self.up_2((x, clip_l2))                 # [(BN) 120 160 256]
        x = self.multiply_fusion_3((x, clip_textuals))  # [(BN) 120 160 256]
        # Fusion: clip | vis => 1 | 1
        x = self.conv_fusion_2((x, vis_1))          # [(BN) 120 160 256]
        x = self.up_3((x, clip_l1))                 # [(BN) 240 320 128]
        # Fusion: clip | vis => 1 | 2
        x = self.conv_fusion_3((x, visual_features))  # [(BN) 240 320 256]
        x = self.up_sample(x)                       # [(BN) 480 640 256]
        return x


class CombineCLIPVisualV2(tf.keras.layers.Layer):
    def __init__(self, name="combine_clip_visual"):
        super().__init__(name=name)

        vis_size = (240, 320)
        filters = 256
        self.level_1 = Level(downscale=1, vis_size=vis_size, filters=filters)
        self.level_2 = Level(downscale=2, vis_size=vis_size, filters=filters)
        self.level_3 = Level(downscale=4, vis_size=vis_size, filters=filters)
        self.level_4 = Level(downscale=8, vis_size=vis_size, filters=filters)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           use_bias=False)

        self.clip_feature_extraction = CLIPFeatureExtraction()
        self.clip_regulizer_loss = tf.keras.losses.CategoricalCrossentropy()

    @tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_features"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        vis = inputs[1]                             # [(BN) 480 640 256]
        clip_features = clip_outputs[0]             # [(BN) 1024]
        clip_1 = clip_outputs[1]                    # [(BN) 56 56 256]
        clip_2 = clip_outputs[2]                    # [(BN) 28 28 512]
        clip_3 = clip_outputs[3]                    # [(BN) 14 14 1024]
        clip_4 = clip_outputs[4]                    # [(BN) 7 7 2048]

        x_level_1 = self.level_1((clip_1, vis))     # [(BN) 240 320 256]
        x_level_2 = self.level_2((clip_2, vis))     # [(BN) 240 320 256]
        x_level_3 = self.level_3((clip_3, vis))     # [(BN) 240 320 256]
        x_level_4 = self.level_4((clip_4, vis))     # [(BN) 240 320 256]

        x = tf.concat(                              # [(BN) 240 320 1024]
            [x_level_1, x_level_2, x_level_3, x_level_4], axis=-1)
        x = self.conv(x)                            # [(BN) 240 320 256]
        clip_features_pred = self.clip_feature_extraction(x)  # [(BN) 1024]
        loss = self.clip_regulizer_loss(clip_features, clip_features_pred)
        # Prevent the layer from ignoring the CLIP features and focus on vis
        self.add_loss(loss)
        return x


class CombineCLIPVisualV1(tf.keras.layers.Layer):
    def __init__(self, name="combine_clip_visual"):
        super().__init__(name=name)

        vis_size = (240, 320)
        filters = 256
        self.level_1 = Level(downscale=1, vis_size=vis_size, filters=filters)
        self.level_2 = Level(downscale=2, vis_size=vis_size, filters=filters)
        self.level_3 = Level(downscale=4, vis_size=vis_size, filters=filters)
        self.level_4 = Level(downscale=8, vis_size=vis_size, filters=filters)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           use_bias=False)
        self.up = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

    @tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_features"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        vis = inputs[1]                             # [(BN) 480 640 256]
        _clip_features = clip_outputs[0]            # [(BN) 1024]
        clip_1 = clip_outputs[1]                    # [(BN) 56 56 256]
        clip_2 = clip_outputs[2]                    # [(BN) 28 28 512]
        clip_3 = clip_outputs[3]                    # [(BN) 14 14 1024]
        clip_4 = clip_outputs[4]                    # [(BN) 7 7 2048]

        x_level_1 = self.level_1((clip_1, vis))     # [(BN) 240 320 256]
        x_level_2 = self.level_2((clip_2, vis))     # [(BN) 240 320 256]
        x_level_3 = self.level_3((clip_3, vis))     # [(BN) 240 320 256]
        x_level_4 = self.level_4((clip_4, vis))     # [(BN) 240 320 256]

        x = tf.concat(                              # [(BN) 240 320 4*256]
            [x_level_1, x_level_2, x_level_3, x_level_4], axis=-1)
        x = self.conv(x)                            # [(BN) 240 320 256]
        x = self.up(x)                              # [(BN) 480 640 256]
        return x


class CombineCLIPVisualV0(tf.keras.layers.Layer):
    def __init__(self, name="combine_clip_visual_legacy"):
        super().__init__(name=name)

        filters = 256
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           use_bias=False)
        self.resize = tf.keras.layers.Resizing(
            240, 320, interpolation='bilinear')
        self.up = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

    @ tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_features"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        visual_features = inputs[1]
        _clip_features = clip_outputs[0]                # [(BN) 1024]
        clip_256 = clip_outputs[1]                      # [(BN) 56 56 256]
        _clip_512 = clip_outputs[2]                     # [(BN) 28 28 512]
        _clip_1024 = clip_outputs[3]                    # [(BN) 14 14 1024]
        _clip_2048 = clip_outputs[4]                    # [(BN) 7 7 2048]

        clip_256r = self.resize(clip_256)               # [(BN) 240 320 256]
        fusion = tf.concat([clip_256r,                  # [(BN) 240 320 2*256]
                            visual_features], axis=-1)
        fusion = self.conv(fusion)                      # [(BN) 240 320 256]
        fusion = self.up(fusion)
        return fusion
