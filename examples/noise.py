from time import time

import tensorflow as tf
from PIL import Image
import numpy as np

from art_utils.noise import get_input_vectors, noise3d, calculate_image
from art_utils.constants import np_grad3, np_perm, np_vertex_table

shape = [512, 512]  # 3840, 2160   for 4K
phases = 1
scaling = 150.0
offset = (0.0, 0.0, 0.0)
v_shape = tf.constant(shape, name='shape')
v_phases = tf.constant(phases, name='phases')
v_scaling = tf.constant(scaling, name='scaling')
v_offset = tf.constant(offset, name='offset')
v_input_vectors = get_input_vectors(v_shape, v_phases, v_scaling, v_offset)
print(v_input_vectors.shape)
perm = tf.constant(np_perm, name='perm')
grad3 = tf.constant(np_grad3, name='grad3')
vertex_table = tf.constant(np_vertex_table, name='vertex_table')
start_time = time()
raw_noise = noise3d(v_input_vectors, perm, grad3, vertex_table, shape[0] * shape[1] * phases)
print(raw_noise.shape)
end_time = time()
print('Time to calculate one iteration: {:.4f}'.format(end_time - start_time))
raw_image_data = calculate_image(raw_noise, phases, v_shape)
input_vectors = get_input_vectors(shape, phases, scaling, offset)
noise = noise3d(input_vectors, np_perm, np_grad3, np_vertex_table, shape[0] * shape[1] * phases)
image_data = calculate_image(noise, phases, shape)
Image.fromarray(image_data.numpy().astype(np.uint8)).show()


# Custom vector repeating the top quarter
base_values = input_vectors[0:int(input_vectors.shape[0] / 4)]
input_vectors = tf.tile(base_values, [4, 1])
noise = noise3d(input_vectors, np_perm, np_grad3, np_vertex_table, shape[0] * shape[1] * phases)
image_data = calculate_image(noise, phases, shape)
Image.fromarray(image_data.numpy().astype(np.uint8)).show()
