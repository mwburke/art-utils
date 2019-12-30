from skimage.draw._polygon2mask import polygon2mask
from skimage.draw import ellipse, circle, polygon
import tensorflow as tf
from PIL import Image
import numpy as np

from art_utils.transform import flip_up_down
from art_utils.math import lerp

img_path = 'img/008.jpg'

img = Image.open(img_path)
img.show()

arr = np.array(img)

center = np.array([2059.5, 3089])
side_length = 500
height =  500 * np.sqrt(3) / 2

vertices = np.array([
    [center[0] - side_length, center[1] - height],
    [center[0] + side_length, center[1] - height],
    [center[0], center[1] + height]
])

flipped_arr = flip_up_down(arr)

ratios = [1, 0.8, 0.6, 0.4, 0.2]

for i, r in enumerate(ratios):
    new_verts = np.zeros(vertices.shape)
    new_verts[:, 0] = lerp(r, vertices[:, 0], center[0])
    new_verts[:, 1] = lerp(r, vertices[:, 1], center[1])
    mask = polygon2mask(arr.shape, new_verts)

    if (i % 2) == 0:
        arr = np.place(arr, mask, flipped_arr)
    else:
        arr = np.place(arr, mask, arr)

new_img = Image.fromarray(arr)
new_img.show()
