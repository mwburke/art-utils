import numpy as np
from PIL import Image
from art_utils.shape import circle_pack
from skimage.draw import circle

x = 1000
y = 2000
max_radius = 200
min_radius = 100
radius_decrease = 50
num_retries = 30


full_mask = np.zeros((x, y, 3)).astype(bool)
num_overlaps = 3
for i in range(num_overlaps):
    circles = circle_pack(x, y, max_radius, min_radius, radius_decrease, num_retries)

    points = np.empty([0, 2])

    for circ in circles:

        rr, cc = circle(circ[0], circ[1], circ[2], shape=[x, y])
        points = np.append(points, np.vstack((rr, cc)).T, axis=0)

    final_points = np.unique(points, axis=0).astype(int)

    full_mask[final_points[:, 0], final_points[:, 1], i] = True

arr = (np.sum(full_mask, axis=2) >= 2).astype(np.uint8) * 255
Image.fromarray(arr).show()
