from time import time

from PIL import Image
import numpy as np

from art_utils.noise import *
from art_utils.constants import np_grad3, np_vertex_table, np_perm

if __name__ == "__main__":
    shape = [512, 512]
    phases = 5
    scaling = 150.0
    offset = (0.0, 0.0, 1.7)
    perm = tf.constant(np_perm, name='perm')
    grad3 = tf.constant(np_grad3, name='grad3')
    vertex_table = tf.constant(np_vertex_table, name='vertex_table')
    input_vectors = get_input_vectors(shape, phases, scaling, offset)
    start_time = time()
    noise = noise3d(input_vectors, np_perm, np_grad3, np_vertex_table, shape[0] * shape[1] * phases)
    end_time = time()
    print('Time to calculate one iteration: {:.4f}'.format(end_time - start_time))
    image_data = calculate_image(noise, phases, shape)
    img = Image.fromarray(image_data.numpy().astype(np.uint8))
    img.show()
