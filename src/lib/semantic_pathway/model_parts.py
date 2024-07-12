#####
# Code taken from https://github.com/cliport
# File: cliport/models/resnet.py
# Apache License 2.0
#####
# Modifications: Conversion to tensorflow
#####

import tensorflow as tf


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        # Layer1
        self.layer1 = tf.keras.Sequential()
        self.layer1.add(tf.keras.layers.Conv2D(
            filters1, kernel_size=1, use_bias=False))
        self.layer1.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())
        self.layer1.add(tf.keras.layers.ReLU())
        # Layer2
        self.layer2 = tf.keras.Sequential()
        self.layer2.add(tf.keras.layers.Conv2D(filters2, kernel_size=kernel_size,
                        dilation_rate=1, strides=strides, padding='same', use_bias=False))
        self.layer2.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())
        self.layer2.add(tf.keras.layers.ReLU())
        # Layer3
        self.layer3 = tf.keras.Sequential()
        self.layer3.add(tf.keras.layers.Conv2D(
            filters3, kernel_size=1, use_bias=False))
        self.layer3.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())

    @tf.function(reduce_retracing=True)
    def call(self, x):
        print("IdentityBlock once")
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += x
        if self.final_relu:
            out = tf.keras.layers.ReLU()(out)
        return out


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm
        filters1, filters2, filters3 = filters
        # Layer1
        self.layer1 = tf.keras.Sequential()
        self.layer1.add(tf.keras.layers.Conv2D(
            filters1, kernel_size=1, use_bias=False))
        self.layer1.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())
        self.layer1.add(tf.keras.layers.ReLU())
        # Layer2
        self.layer2 = tf.keras.Sequential()
        self.layer2.add(tf.keras.layers.Conv2D(filters2, kernel_size=kernel_size,
                        dilation_rate=1, strides=strides, padding='same', use_bias=False))
        self.layer2.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())
        self.layer2.add(tf.keras.layers.ReLU())
        # Layer3
        self.layer3 = tf.keras.Sequential()
        self.layer3.add(tf.keras.layers.Conv2D(
            filters3, kernel_size=1, use_bias=False))
        self.layer3.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())
        # Shortcut
        self.shortcut = tf.keras.Sequential()
        self.shortcut.add(tf.keras.layers.Conv2D(
            filters3, kernel_size=1, strides=strides, use_bias=False))
        self.shortcut.add(tf.keras.layers.BatchNormalization(
            epsilon=1e-5) if self.batchnorm else tf.keras.layers.Identity())

    @tf.function(reduce_retracing=True)
    def call(self, x):
        print("ConvBlock once")
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += self.shortcut(x)
        if self.final_relu:
            out = tf.keras.layers.ReLU()(out)
        return out
