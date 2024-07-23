import os

import tensorflow as tf
from einops import rearrange
from lib.mvnerf.layers import ResNetMLPBlock, Readout


class GraspReadout(tf.keras.Model):
    def __init__(self, use_bias=False, **kwargs):
        super(GraspReadout, self).__init__(**kwargs)
        activation_ds = 64
        self.activation_downscale_1 = tf.keras.layers.Dense(activation_ds, activation='elu',
                                                            kernel_initializer='he_normal')
        self.activation_downscale_2 = tf.keras.layers.Dense(activation_ds, activation='elu',
                                                            kernel_initializer='he_normal')
        self.activation_downscale_3 = tf.keras.layers.Dense(activation_ds, activation='elu',
                                                            kernel_initializer='he_normal')
        self.activation_downscale_4 = tf.keras.layers.Dense(activation_ds, activation='elu',
                                                            kernel_initializer='he_normal')

        self.combined_activation_downscale = tf.keras.layers.Dense(
            64, activation='elu')

        self.readout = tf.keras.Sequential(
            [ResNetMLPBlock(128, 64, transform_shortcut=True, activation='elu', kernel_initializer='he_normal'),
             ResNetMLPBlock(64, 64, activation='elu',
                            kernel_initializer='he_normal'),
             Readout(1, use_bias=use_bias, kernel_initializer='he_normal')])

    def call(self, inputs, *args, **kwargs):
        ds_activation_1 = self.activation_downscale_1(inputs[0])
        ds_activation_2 = self.activation_downscale_2(inputs[1])
        ds_activation_3 = self.activation_downscale_3(inputs[2])
        ds_activation_4 = self.activation_downscale_4(inputs[3])
        combined_activations = tf.concat(
            [ds_activation_1, ds_activation_2, ds_activation_3, ds_activation_4], axis=-1)
        combined_activations = self.combined_activation_downscale(
            combined_activations)
        combined_activations = rearrange(
            combined_activations, 'b np n5 d -> b np (n5 d)')
        output = self.readout(combined_activations)[..., 0]
        return output


class Tile(tf.keras.Model):
    def __init__(self, shape=(None, 240, 320, 256), name="tile", ):
        super().__init__(name=name)

        self.shape = shape
        self.dense = tf.keras.layers.Dense(units=self.shape[3],
                                           use_bias=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_textuals")])
    def call(self, inputs):
        x = self.dense(inputs)
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # [(BN) 1 1 256]
        x = tf.repeat(x, repeats=self.shape[1], axis=1)  # [(BN) 240 1 256]
        x = tf.repeat(x, repeats=self.shape[2], axis=2)  # [(BN) 240 320 256]
        return x
