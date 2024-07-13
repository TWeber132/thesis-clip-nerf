#####
# Code taken from https://github.com/cliport
# File: cliport/models/core/fusion.py
# Apache License 2.0
#####
# Modifications: Conversion to tensorflow
#####

import tensorflow as tf
import numpy as np
from typing import Tuple


class Fusion(tf.keras.layers.Layer):

    def __init__(self, x2_proj: tf.keras.layers.Dense = None):
        super().__init__()
        self.x2_proj = x2_proj

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32, name="image"),
                                  tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="text")])
    def tile_x2(self, x1, x2):
        if self.x2_proj:
            x2 = self.x2_proj(x2)
        # Instead of torch:  x2 = tf.expand_dims(tf.expand_dims(x2, axis=-1), axis=-1)
        # And since tensorflow works in NHWC manner instead of NCHW like torch we would have to do:
        # x2 = tf.expand_dims(tf.expand_dims(x2, axis=1), axis=1)
        x2 = tf.expand_dims(tf.expand_dims(x2, axis=1), axis=1)
        # Instead of torch:  x2 = x2.repeat(x1.shape[0], 1, x1.shape[-2], x1.shape[-1])
        # which effecively is: x2 = x2.repeat(x1.shape[0], 1, x1.shape[2], x1.shape[3])
        #                                            ax0,       ax1,        ax2,            ax3
        # And since tensorflow works in NHWC manner instead of NCHW like torch we would have to do:
        # x2 = x2.repeat(x1.shape[0], x1.shape[1], x1.shape[2], 1)
        x2 = tf.repeat(x2, repeats=tf.shape(x1)[0], axis=0)
        x2 = tf.repeat(x2, repeats=tf.shape(x1)[1], axis=1)
        x2 = tf.repeat(x2, repeats=tf.shape(x1)[2], axis=2)
        return x2

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32, name="image"),
                                   tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="text"))])
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        raise NotImplementedError()


class FusionMult(Fusion):
    """ x1 * x2 """

    def __init__(self, text_proj: tf.keras.layers.Dense = None):
        super(FusionMult, self).__init__(text_proj=text_proj)

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32, name="image"),
                                   tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="text"))])
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x1, x2 = inputs
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2)
        return x1 * x2


class FusionAdd(Fusion):
    """ x1 + x2 """

    def __init__(self, x2_proj: tf.keras.layers.Dense = None):
        super(FusionAdd, self).__init__(x2_proj=x2_proj)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x1, x2 = inputs
        if tf.shape(x1) != tf.shape(x2) and len(tf.shape(x1)) != len(tf.shape(x2)):
            x2 = self.tile_x2(x1, x2)
        return x1 + x2


class FusionConv(Fusion):
    """ 1x1 convs after [x1; x2] """

    def __init__(self, x2_proj: tf.keras.layers.Dense = None, filters: int = 3):
        super(FusionConv, self).__init__(x2_proj=x2_proj)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=1,
                                   use_bias=False)])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x1, x2 = inputs
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2)
        x = tf.concat([x1, x2], axis=-1)    # [B, H, W, 2C]
        x = self.conv(x)                    # [B, H, W, C]
        return x
