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

    def __init__(self, filters, mid_filters=None, name="double_conv"):
        super().__init__(name=name)
        if not mid_filters:
            mid_filters = filters
        self.conv_1 = tf.keras.layers.Conv2D(
            mid_filters, kernel_size=3, padding='same', use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.relu_1 = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding='same', use_bias=False)
        self.bn_2 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.relu_2 = tf.keras.layers.ReLU()

    @tf.function(reduce_retracing=True)
    def call(self, x):
        tf.print("tracing ... ")
        print("DoubleConv once")
        x = self.relu_1(self.bn_1(self.conv_1(x)))
        x = self.relu_2(self.bn_2(self.conv_2(x)))
        return x


class Up(tf.keras.layers.Layer):
    """Upscaling then double conv"""

    def __init__(self, filters, in_filters, name="up"):
        super().__init__(name=name)

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')  # align_corners
        self.conv = DoubleConv(filters, in_filters // 2)

    @tf.function(reduce_retracing=True)
    def call(self, x):
        x = self.up(x)
        return self.conv(x)


# class Up(tf.keras.layers.Layer):
#     """Upscaling then double conv"""

#     def __init__(self, filters, in_filters, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = tf.keras.layers.UpSampling2D(
#                 size=2, interpolation='bilinear')  # align_corners
#             self.conv = DoubleConv(filters, in_filters // 2)
#         else:
#             self.up = tf.keras.layers.ConvTranspose2d(
#                 in_filters // 2, kernel_size=2, strides=2)
#             self.conv = DoubleConv(filters)

#     @tf.function(reduce_retracing=True)
#     def call(self, inputs):
#         x1, x2 = inputs
#         x1 = self.up(x1)
#         # input was CHW (torch) and now has to be HWC because of tensorflow
#         # diffY = x2.shape[2] - x1.shape[2]
#         # diffX = x2.shape[3] - x1.shape[3]
#         # becomes:
#         diffY = tf.shape(x2)[1] - tf.shape(x1)[1]
#         diffX = tf.shape(x2)[2] - tf.shape(x1)[2]
#         x1 = tf.pad(x1, [[0, 0], [diffX // 2, diffX - diffX // 2],
#                          [diffY // 2, diffY - diffY // 2], [0, 0]])
#         # Concat the 1024*x features
#         x = tf.concat([x2, x1], axis=-1)
#         return self.conv(x)


class Down(tf.keras.layers.Layer):
    """Downscaling with maxpool then double conv"""

    def __init__(self, filters, name="down"):
        super().__init__(name=name)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2)
        self.double_conv = DoubleConv(filters)

    @tf.function(reduce_retracing=True)
    def call(self, x):
        x = self.maxpool(x)
        return self.double_conv(x)
