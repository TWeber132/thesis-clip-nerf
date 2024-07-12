#####
# Code inspired by https://github.com/cliport
# File: cliport/models/clip_lingunet.py
# Apache License 2.0
#####
# Modifications:  Conversion to tensorflow
#####

from unet.model_parts import Up
from semantic_pathway.fusion import FusionMult
from semantic_pathway.model_parts import ConvBlock, IdentityBlock

from typing import Union, List, Tuple
import tensorflow as tf


class SemanticPathway(tf.keras.Model):
    output_dim: int
    up_factor: int
    bilinear: bool
    batchnorm: bool

    clip_model: tf.keras.Model
    decoder_model: tf.keras.Model

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 32
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.batchnorm = True

        self._init_clip()
        self._init_decoder()

    def _init_clip(self) -> None:
        clip_dir = "/home/robot/docker_volume/nn_model/clip/model_weights"
        self.clip_model = tf.keras.models.load_model(clip_dir)
        self.clip_model.trainable = False

    def _init_decoder(self) -> None:
        # Text
        self.text_proj1 = tf.keras.layers.Dense(1024)
        self.text_proj2 = tf.keras.layers.Dense(512)
        self.text_proj3 = tf.keras.layers.Dense(256)
        self.text_fuser1 = FusionMult(text_proj=self.text_proj1)
        self.text_fuser2 = FusionMult(text_proj=self.text_proj2)
        self.text_fuser3 = FusionMult(text_proj=self.text_proj3)
        # Vision
        self.conv1 = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=3, strides=1, padding="same", use_bias=False, activation='relu')
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)
        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)
        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)
        self.layer1 = tf.keras.Sequential([
            ConvBlock(filters=[64, 64, 64], kernel_size=3,
                      strides=1, batchnorm=self.batchnorm),
            IdentityBlock(filters=[64, 64, 64], kernel_size=3,
                          strides=1, batchnorm=self.batchnorm),
            tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        ])
        self.layer2 = tf.keras.Sequential([
            ConvBlock(filters=[32, 32, 32], kernel_size=3,
                      strides=1, batchnorm=self.batchnorm),
            IdentityBlock(filters=[32, 32, 32], kernel_size=3,
                          strides=1, batchnorm=self.batchnorm),
            tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear'),
        ])
        self.layer3 = tf.keras.Sequential([
            ConvBlock(filters=[16, 16, 16], kernel_size=3,
                      strides=1, batchnorm=self.batchnorm),
            IdentityBlock(filters=[16, 16, 16], kernel_size=3,
                          strides=1, batchnorm=self.batchnorm),
            tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear'),
        ])
        self.conv2 = tf.keras.layers.Conv2D(self.output_dim, kernel_size=1)
        self.final_conv = tf.keras.layers.Conv2D(
            filters=1, kernel_size=7, padding='same', use_bias=False)

        return

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32, name="image")])
    def _encode_image(self, image) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        print("SemanticPathway image encode once")
        outputs = self.clip_model.encode_image(image)
        output, x_layer1, x_layer2, x_layer3, x_layer4 = outputs
        # We only care about the intermediate features and drop image clip embedding
        return x_layer1, x_layer2, x_layer3, x_layer4

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 77), dtype=tf.int32, name="text")])
    def _encode_text(self, text) -> tf.Tensor:
        print("SemanticPathway text encode once")
        # text = tf.squeeze(text, axis=0)
        output = self.clip_model.encode_text(text)
        return output

    @tf.function(input_signature=[(tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32, name="image"),
                                   tf.TensorSpec(shape=(1, 77), dtype=tf.int32, name="text"))])
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        print("SemanticPathway once")
        image_nhwc, text = inputs
        # Pytorch would have handled image and feature maps as "channel_first" but
        # the whole tf clip model is build in tensorflows default "channel_last"
        # image_nchw = tf.keras.layers.Permute(dims=(3, 1, 2))(image_nhwc)
        # Encode
        a_layer1, a_layer2, a_layer3, a_layer4 = self._encode_image(image_nhwc)
        text_embedding = self._encode_text(text)
        # Decode
        x = self.conv1(a_layer4)
        x = self.text_fuser1((x, text_embedding))
        x = self.up1((x, a_layer3))
        x = self.text_fuser2((x, text_embedding))
        x = self.up2((x, a_layer2))
        x = self.text_fuser3((x, text_embedding))
        x = self.up3((x, a_layer1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = tf.keras.layers.Resizing(height=tf.shape(image_nhwc)[1], width=tf.shape(
            image_nhwc)[2], interpolation='bilinear')(x)
        x = self.final_conv(x)
        return x
