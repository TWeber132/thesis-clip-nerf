#####
# Code taken from https://github.com/milesial/Pytorch-UNet
# File: unet/unet_parts.py
# GNU General Public License v3.0
#####
# Modifications: Conversion to tensorflow
#####

import tensorflow as tf


class DoubleConv(tf.keras.layers.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                mid_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                out_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU()])

    @tf.function(reduce_retracing=True)
    def call(self, x):
        print("DoubleConv once")
        return self.double_conv(x)


class Up(tf.keras.layers.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = tf.keras.layers.UpSampling2D(
                size=2, interpolation='bilinear')  # align_corners
            self.conv = DoubleConv(out_channels, in_channels // 2)
        else:
            self.up = tf.keras.layers.ConvTranspose2d(
                in_channels // 2, kernel_size=2, strides=2)
            self.conv = DoubleConv(out_channels)

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        print("Up once")
        image, a_layer_x = inputs
        image = self.up(image)
        # input was CHW (torch) and now has to be HWC because of tensorflow
        # diffY = x2.shape[2] - x1.shape[2]
        # diffX = x2.shape[3] - x1.shape[3]
        # becomes:
        diffY = tf.shape(a_layer_x)[1] - tf.shape(image)[1]
        diffX = tf.shape(a_layer_x)[2] - tf.shape(image)[2]
        image = tf.pad(image, [[0, 0], [diffX // 2, diffX - diffX // 2],
                               [diffY // 2, diffY - diffY // 2], [0, 0]])
        # Concat the 1024*x features
        x = tf.concat([a_layer_x, image], axis=-1)
        return self.conv(x)
