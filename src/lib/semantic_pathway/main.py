from nn_model.clip.utils import preprocess, tokenize, _denormalize
from nn_model.semantic_pathway.model import SemanticPathway

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')

if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)

    _image = Image.open("/home/robot/docker_volume/nn_training/lego.jpg")
    _image2 = Image.open("/home/robot/docker_volume/nn_training/lego2.jpeg")
    _text = ["a yellow lego bulldozer"]
    image = preprocess(_image)
    image2 = preprocess(_image2)
    text = tokenize(_text)

    semantic_model = SemanticPathway()
    x = semantic_model((image, text))
    x2 = semantic_model((image2, text))
    x = x.numpy()
    x = np.squeeze(x, axis=0)
    x2 = x2.numpy()
    x2 = np.squeeze(x2, axis=0)
    x = _denormalize(x)
    x2 = _denormalize(x2)
    plt.imshow(x)
    plt.show()
    plt.imshow(x2)
    plt.show()
