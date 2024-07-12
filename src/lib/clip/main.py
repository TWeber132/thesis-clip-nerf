from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

from .utils import preprocess, tokenize


def load_clip():
    clip_dir = "/home/jovyan/data/storage/clip_weights"
    clip = tf.keras.models.load_model(clip_dir)
    return clip


def main():
    # tf.config.run_functions_eagerly(True)
    # tf.compat.v1.disable_eager_execution()
    # print("eager:", tf.executing_eagerly())

    _image = Image.open("/home/robot/docker_volume/nn_training/lego.jpg")
    _image2 = Image.open("/home/robot/docker_volume/nn_training/lego2.jpeg")
    _texts = ["a excavator", "an excavator",
              "a yellow lego excavator", "a yellow lego bulldozer"]
    image = preprocess(_image)
    image2 = preprocess(_image2)
    texts = tokenize(_texts)

    clip = load_clip()

    logits_per_image, logits_per_text = clip.predict((image, texts))
    tf_probs = tf.nn.softmax(logits_per_image, axis=1)
    tf_probs = np.array(tf_probs)
    print(tf_probs)
    logits_per_image, logits_per_text = clip.predict((image2, texts))
    tf_probs = tf.nn.softmax(logits_per_image, axis=1)
    tf_probs = np.array(tf_probs)
    print(tf_probs)

    outputs = clip.encode_image(image)
    print("Vison model output:", outputs[0].shape)
    print("Vison model activation layer1:", outputs[1].shape)
    print("Vison model activation layer2:", outputs[2].shape)
    print("Vison model activation layer3:", outputs[3].shape)
    print("Vison model activation layer4:", outputs[4].shape)

    texts = tf.squeeze(texts, axis=0)
    outputs = clip.encode_text(texts)
    print("Transformer model output", outputs.shape)
    return


if __name__ == '__main__':
    main()
