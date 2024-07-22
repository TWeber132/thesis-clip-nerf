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

        self.combined_activation_downscale = tf.keras.layers.Dense(64, activation='elu')

        self.readout = tf.keras.Sequential(
            [ResNetMLPBlock(128, 64, transform_shortcut=True, activation='elu', kernel_initializer='he_normal'),
             ResNetMLPBlock(64, 64, activation='elu', kernel_initializer='he_normal'),
             Readout(1, use_bias=use_bias, kernel_initializer='he_normal')])

    def call(self, inputs, *args, **kwargs):
        ds_activation_1 = self.activation_downscale_1(inputs[0])
        ds_activation_2 = self.activation_downscale_2(inputs[1])
        ds_activation_3 = self.activation_downscale_3(inputs[2])
        ds_activation_4 = self.activation_downscale_4(inputs[3])
        combined_activations = tf.concat([ds_activation_1, ds_activation_2, ds_activation_3, ds_activation_4], axis=-1)
        combined_activations = self.combined_activation_downscale(combined_activations)
        combined_activations = rearrange(combined_activations, 'b np n5 d -> b np (n5 d)')
        output = self.readout(combined_activations)[..., 0]
        return output
