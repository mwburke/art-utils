import tensorflow as tf

@tf.function
def simple_nn_layer(x, y):
  return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3000, 3000))
y = tf.random.uniform((3000, 3000))

simple_nn_layer(x, y)