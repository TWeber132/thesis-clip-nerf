from typing import Any
import tensorflow as tf


def load_clip():
    clip_dir = "/home/jovyan/data/storage/clip_weights"
    clip = tf.keras.models.load_model(clip_dir)
    return clip


class CLIP(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.clip = load_clip()
        self.clip.trainable = False
        
    def call(self, inputs):
        return self.clip(inputs)


class CLIPVisualEncoder(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.clip = load_clip()
        self.clip.trainable = False

    def call(self, inputs):
        return self.clip.encode_image(inputs)


class CLIPTextualEncoder(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.clip = load_clip()
        self.clip.trainable = False

    def call(self, inputs):
        return self.clip.encode_text(inputs)
