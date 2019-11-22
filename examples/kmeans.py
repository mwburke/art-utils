import tensorflow as tf
from PIL import Image
import numpy as np

from art_utils.kmeans import kmeans

# Probably can downsample actual points to do kmeans on
# Then do assignment of color based on nearest centroid like in the algorithm

img_file = '004'
img_path = f'img/{img_file}.jpg'

img = Image.open(img_path)

data = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
orig_shape = data.shape

print(data.shape, data.shape[0] * data.shape[1])

data = tf.reshape(data, [data.shape[0] * data.shape[1], 3])

print(data.shape)

k = 7
n_iter = 10

assignments, centroids = kmeans(data, k, n_iter)

print(centroids)

data = data.numpy()

temp1 = data
temp2 = data

for i in range(k):
    temp1[assignments == i, :] = centroids[i, :]

out_data = tf.reshape(temp1, orig_shape).numpy().astype(np.uint8)

Image.fromarray(out_data).show()

# Uncomment this to get a strip with cluser colors appended to bottom of image
width = out_data.shape[1]

color_data = np.zeros([500, width, 3], dtype=np.uint8)
for i in range(k):
    color_data[:, int(width * i / k):int(width * (i + 1) / k), :] = centroids[i, :].numpy().astype(np.uint8).reshape([1, 1, 3])

out_data = tf.reshape(temp2, orig_shape).numpy().astype(np.uint8)
out_data = np.concatenate([out_data, color_data], axis=0)

print(out_data.shape)

Image.fromarray(out_data).show()
