import tensorflow as tf
import numpy as np
from .nerf_utils import position_encoding
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
            tf.keras.layers.Conv2D(n_features // 2, kernel_size=1, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.conv_features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=7, padding='same', use_bias=False, strides=2),
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
            tf.keras.layers.Dense(embed_dim * mlp_ratio, activation='gelu', name=f'{self.name}/dense_0'),
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
        self.cls_token = tf.Variable(tf.zeros([1, 1, embed_dim]), trainable=True, name='cls_token')
        embed_len = num_patches + 1
        self.pos_embedding = tf.Variable(tf.random.normal([1, embed_len, embed_dim]) * 0.02, trainable=True,
                                         name='pos_embedding')

        # TODO not really general ... need refactoring
        feature_hooks = [hooks[0]] + [hooks[i] - hooks[i - 1] for i in range(1, len(hooks))]
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

        self.vit = VisionTransformer(img_size, patch_size, embed_dim, mlp_ratio, num_classes, hooks)
        self.post_process_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(features[0], kernel_size=1),
            tf.keras.layers.Conv2DTranspose(features[0], kernel_size=4, strides=4)
        ])
        self.post_process_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(features[1], kernel_size=1),
            tf.keras.layers.Conv2DTranspose(features[1], kernel_size=2, strides=2)
        ])
        self.post_process_3 = tf.keras.layers.Conv2D(features[2], kernel_size=1)
        self.post_process_4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(features[3], kernel_size=1),
            tf.keras.layers.Conv2D(features[3], kernel_size=3, strides=2, padding='same')
        ])

        self.conv_decode_1 = tf.keras.layers.Conv2D(n_features, kernel_size=3, padding='same', use_bias=False)
        self.conv_decode_2 = tf.keras.layers.Conv2D(n_features, kernel_size=3, padding='same', use_bias=False)
        self.conv_decode_3 = tf.keras.layers.Conv2D(n_features, kernel_size=3, padding='same', use_bias=False)
        self.conv_decode_4 = tf.keras.layers.Conv2D(n_features, kernel_size=3, padding='same', use_bias=False)

        self.upsample_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_2 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upsample_3 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')
        self.upsample_4 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')

        self.output_conv = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_features, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_features // 2, kernel_size=3, padding='same')
        ])

    def call(self, inputs, training=False, mask=None):
        _, features = self.vit(inputs)
        features = [rearrange(f[:, 1:], 'b (h w) c -> b h w c', h=self.vit.grid_size[0], w=self.vit.grid_size[1]) for f
                    in features]
        latents = tf.concat([
            self.upsample_1(self.conv_decode_1(self.post_process_1(features[0]))),
            self.upsample_2(self.conv_decode_2(self.post_process_2(features[1]))),
            self.upsample_3(self.conv_decode_3(self.post_process_3(features[2]))),
            self.upsample_4(self.conv_decode_4(self.post_process_4(features[3])))
        ], axis=-1)
        latents = self.output_conv(latents)
        return latents


class VisualFeatures(tf.keras.Model):
    def __init__(self, n_features=256, original_image_size=(480, 640), **kwargs):
        super(VisualFeatures, self).__init__(**kwargs)

        self.conv_features = ConvolutionalEncoder(n_features=n_features)

        self.vision_transformer = VisionTransformerEncoder(n_features=n_features)

        transformer_image_size = (224, 224)
        self.transformer_downscale = tf.keras.layers.Resizing(transformer_image_size[0], transformer_image_size[1],
                                                              interpolation='bilinear')
        self.transformer_upscale = tf.keras.layers.Resizing(original_image_size[0] // 2, original_image_size[1] // 2,
                                                            interpolation='bilinear')

        self.feature_upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

    def call(self, inputs, training=False, mask=None):
        # b h w c
        image_transformer = self.transformer_downscale(inputs)
        latents = self.vision_transformer(image_transformer)
        # scale image back to half the original size
        latents = self.transformer_upscale(latents)
        skip_latents = self.conv_features(inputs)
        features = tf.concat([latents, skip_latents], axis=-1)
        features = self.feature_upsample(features)
        return features


class ResNetMLPBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, transform_shortcut=False, activation='relu',
                 kernel_initializer='glorot_uniform', **kwargs):
        super(ResNetMLPBlock, self).__init__(**kwargs)

        self.layer_0 = tf.keras.layers.Dense(hidden_size, kernel_initializer=kernel_initializer)
        self.layer_1 = tf.keras.layers.Dense(output_size, kernel_initializer=kernel_initializer)

        if activation == 'relu':
            self.prelu_0 = tf.keras.layers.ReLU()
            self.prelu_1 = tf.keras.layers.ReLU()
        elif activation == 'elu':
            self.prelu_0 = tf.keras.layers.ELU()
            self.prelu_1 = tf.keras.layers.ELU()
        else:
            raise ValueError(f'activation {activation} not supported')

        if transform_shortcut:
            self.shortcut = tf.keras.layers.Dense(output_size, use_bias=False, kernel_initializer=kernel_initializer)
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
        encoded_pos = position_encoding(inputs[0], self.n_freq, self.pos_encoding_freq)
        if self.embed_direction_vector:
            encoded_dir = position_encoding(inputs[1], self.n_freq, self.pos_encoding_freq)
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
        encoded_pos = position_encoding(inputs[0], self.n_freq, self.pos_encoding_freq)
        if self.embed_direction_vector:
            encoded_dir = position_encoding(inputs[1], self.n_freq, self.pos_encoding_freq)
        else:
            encoded_dir = inputs[1]
        x = tf.concat([encoded_pos, encoded_dir, inputs[2]], axis=-1)
        x = self.layer_0(x)
        outputs = [x]
        for block in self.feature_blocks:
            outputs.append(block(outputs[-1]))

        pre_fusion = rearrange(outputs[-1], '(b nv) nr np d -> b nv nr np d', nv=self.n_views)
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

        self.softplus = tf.keras.layers.Activation(tf.keras.activations.softplus)
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
        self.output_layer = tf.keras.layers.Dense(output_size, use_bias=use_bias, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False, mask=None):
        output = self.prelu(inputs)
        output = self.output_layer(output)
        return output
