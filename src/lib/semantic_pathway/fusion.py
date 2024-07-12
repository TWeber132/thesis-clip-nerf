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

    def __init__(self, text_proj: tf.keras.layers.Dense = None):
        super().__init__()
        self.text_proj = text_proj

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32, name="image"),
                                  tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="text")])
    def tile_text(self, image, text):
        print("Fusion tile_text once")
        if self.text_proj:
            text = self.text_proj(text)
        # Instead of torch:  text = tf.expand_dims(tf.expand_dims(text, axis=-1), axis=-1)
        # And since tensorflow works in NHWC manner instead of NCHW like torch we would have to do:
        # text = tf.expand_dims(tf.expand_dims(text, axis=1), axis=1)
        text = tf.expand_dims(tf.expand_dims(text, axis=1), axis=1)
        # Instead of torch:  text = text.repeat(image.shape[0], 1, image.shape[-2], image.shape[-1])
        # which effecively is: text = text.repeat(image.shape[0], 1, image.shape[2], image.shape[3])
        #                                            ax0,       ax1,        ax2,            ax3
        # And since tensorflow works in NHWC manner instead of NCHW like torch we would have to do:
        # text = text.repeat(image.shape[0], image.shape[1], image.shape[2], 1)
        text = tf.repeat(text, repeats=tf.shape(image)[0], axis=0)
        text = tf.repeat(text, repeats=tf.shape(image)[1], axis=1)
        text = tf.repeat(text, repeats=tf.shape(image)[2], axis=2)
        return text

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32, name="image"),
                                   tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="text"))])
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        print("Fusion notimpl once")
        raise NotImplementedError()


class FusionMult(Fusion):

    def __init__(self, text_proj: tf.keras.layers.Dense = None):
        super(FusionMult, self).__init__(text_proj=text_proj)

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32, name="image"),
                                   tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="text"))])
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        print("FusionMult once")
        image, text = inputs
        if image.shape != text.shape and len(image.shape) != len(text.shape):
            text = self.tile_text(image, text)
        return image * text
