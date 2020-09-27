from art_utils.shape import CuttablePolygon, split_polygons
from art_utils.shape import *
import numpy as np
from PIL import Image

points = np.array([[100, 200], [100, 100], [200, 100], [200, 200]])

polygon = CuttablePolygon(points)
polygons = [polygon]

# cut_lines = [[np.array([150, 201]), np.array([150, 99])]]
# cut_lines = [[np.array([100, 98]), np.array([200, 202])]]
# cut_lines = [[np.array([150, 201]), np.array([150, 99])], [np.array([100, 98]), np.array([200, 202])]]
# cut_lines = [[np.array([100, 98]), np.array([200, 202])], [np.array([150, 201]), np.array([150, 99])]]
cut_lines = [[np.array([201, 150]), np.array([99, 150])], [np.array([150, 201]), np.array([150, 99])]]

cut_polygons = split_polygons(polygons, cut_lines)

# print([c.points for c in cut_polygons])
#
# print([c.get_shrunken_points(0.8) for c in cut_polygons])


arr = np.ones((200, 200)) * 255
arr2 = np.zeros((200, 200))

mask = np.zeros((200, 200)).astype(bool)

for poly in cut_polygons:
    mask_poly_points = poly.get_shrunken_points(0.8)
    mask_poly_points[:, 1] = 200 - mask_poly_points[:, 1]

    shape_mask = get_polygon_fill_mask(mask_poly_points, (200, 200, 3))
    mask += shape_mask

final_arr = arr * mask + arr2 * (1 - mask)

Image.fromarray(final_arr.astype(np.uint8)).show()
