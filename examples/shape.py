import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

from art_utils.shape import chiakins_curve, get_polygon_fill_mask


points = np.array([[0.05, 20], [0.25 ,360], [0.75, 460], [0.95, 20]])

iterations = 1
updated_points = chiakins_curve(points, iterations)
plt.plot(updated_points[:, 0], updated_points[:, 1], label='1 iteration')


iterations = 2
updated_points = chiakins_curve(points, iterations)
plt.plot(updated_points[:, 0], updated_points[:, 1], label='2 iterations')


iterations = 3
updated_points = chiakins_curve(points, iterations)
plt.plot(updated_points[:, 0], updated_points[:, 1], label='3 iterations')
plt.legend()
plt.show()


arr = np.zeros((500, 500))

points = np.array([
    [100, 100],
    [100, 300],
    # [300, 300],
    [300, 100]
])

mask = get_polygon_fill_mask(points, arr.shape)


arr[mask] = 255


im = Image.fromarray(np.uint8(arr))
im.show()