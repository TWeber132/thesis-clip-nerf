import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=3, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = self.get_valid_indices()
        self.on_epoch_end()

    def get_valid_indices(self):
        return np.arange(len(self.dataset))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.indices[index *
                             self.batch_size:(index + 1) * self.batch_size]
        ret = self.get_data(batch)
        return ret

    def get_data(self, batch):
        raise NotImplementedError(
            "This method should be implemented in a subclass.")
