from typing import Any
import tensorflow as tf


def load_clip():
    clip_dir = "/home/jovyan/data/storage/clip_weights"
    clip = tf.keras.models.load_model(clip_dir)
    return clip


class CLIP():
    def __init__(self) -> None:
        self = load_clip()
        self.trainable = False


class CLIPVisualEncoder():
    def __init__(self) -> None:
        self = load_clip().visual
        self.trainable = False


class CLIPTextualEncoder():
    def __init__(self) -> None:
        self.clip = load_clip()
        self.trainable = False

    def call(self, inputs):
        self.clip.encode_text(inputs)
