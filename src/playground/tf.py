import tensorflow as tf

random_tensor = tf.random.uniform([2, 4])

dense = tf.keras.layers.Dense(5)

output = dense(random_tensor)

print(output)
