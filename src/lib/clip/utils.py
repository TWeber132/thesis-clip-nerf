import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Any, Tuple, List, Union
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt

from .simple_tokenizer import SimpleTokenizer


def preprocess(image: np.ndarray, to_size: int = 224) -> np.ndarray:
    image = Image.fromarray(image)
    image = _resize(image, size=to_size)
    image = _center_crop(image)
    # We dont need any depth information from the image
    image.convert("RGB")
    img_np = np.asarray(image, dtype=np.uint8)
    img_norm = _normalize(img_np)
    # _ = ._denormalize(img_norm)

    img_norm = np.expand_dims(img_norm, axis=0)
    return img_norm


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32, name="image")])
def preprocess_tf(images: tf.Tensor, to_size: int = 224, normalize: bool = False) -> tf.Tensor:
    # Resize and crop
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    if w > h:
        images = tf.image.resize(
            images, [int(to_size * w / h), to_size], method='bicubic')
    else:
        images = tf.image.resize(
            images, [to_size, int(to_size * w / h)], method='bicubic')
    images = tf.image.resize_with_crop_or_pad(images, to_size, to_size)

    # Normalize
    if normalize:
        images = tf.math.divide(images, 255.0)

    # Standardize
    color_mean = tf.constant(
        [0.48145466, 0.4578275, 0.40821073], dtype=tf.float32)
    color_std = tf.constant(
        [0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)

    images = tf.math.subtract(images, color_mean)
    images = tf.math.divide(images, color_std)
    return images


def _resize(img: Image.Image, size: int = None) -> Image.Image:
    if size is None:
        return img

    w, h = img.size
    if w > h:
        img_resized = img.resize(
            (int(size * w / h), size), resample=Image.Resampling.BICUBIC)
    else:
        img_resized = img.resize(
            (size, int(size * h / w)), resample=Image.Resampling.BICUBIC)
    return img_resized


def _center_crop(img: Image.Image, size: int = None):
    if size is None:
        size = min(img.width, img.height)

    left = np.floor((img.width - size) / 2)
    upper = np.floor((img.height - size) / 2)
    right = left + size
    lower = upper + size
    img_cropped = img.crop((left, upper, right, lower))
    return img_cropped


def _normalize(img: np.ndarray) -> np.ndarray:
    # Given by CLIP
    color_mean = np.asarray(
        [0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    color_std = np.asarray(
        [0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    # Normalize
    img_normalized = np.ndarray(shape=(img.shape), dtype=np.float32)
    img_normalized = ((img / np.float32(255)) - color_mean) / color_std
    return img_normalized


def _denormalize(img: np.ndarray) -> np.ndarray:
    # Given by CLIP
    color_mean = np.asarray(
        [0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    color_std = np.asarray(
        [0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    # Denormalize
    img_denormalized = np.ndarray(shape=(img.shape), dtype=np.uint8)
    img_denormalized = np.uint8(((img * color_std) + color_mean) * 255)
    return img_denormalized


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> np.ndarray:
    #####
    # Code taken from https://github.com/openai/CLIP
    # File: clip/clip.py
    # MIT License
    #####
    # Modifications: _Tokenizer -> SimpleTokenizer
    #####
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    A tf.Tensor of dtype= is returned.
    """

    _tokenizer = SimpleTokenizer()

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] +
                  _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = tokens

    # result = np.expand_dims(result, axis=0)
    return result
