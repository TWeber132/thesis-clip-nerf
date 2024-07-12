import tensorflow as tf
import numpy as np
from PIL import Image
from clip.utils import preprocess, _denormalize
import matplotlib.pyplot as plt

rand = tf.random.normal([1, 8, 8, 3], dtype=tf.float32).numpy()


def test(o):
    for step in range(7):
        o = tf.keras.Sequential(tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear'))(o)
    return o


out = test(rand)
out.numpy()
out = np.squeeze(out, axis=0)
plt.imshow(out)
plt.show()
