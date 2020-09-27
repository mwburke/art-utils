# TODO: circle rasterization
# TODO: line intersection
# TODO: point in polygon
# TODO: marching squares

from art_utils.constants import PI

from typing import List
from random import random
from numpy import array, pi, cos, sin, round, append, arange, abs, floor, arccos, dot, clip
from numpy import arctan2, mean, argsort, empty, empty_like, ones, unique, append
from numpy import meshgrid, vstack, arange, squeeze, atleast_2d, zeros, empty
from numpy.linalg import norm
from skimage.draw import polygon2mask, circle


def degrees_to_radians(degrees):
    return degrees / 180 * PI


def chiakins_curve(coords: array, refinements: int=5) -> array:

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def angle_between(points: array) -> array:
    angles = arctan2(points[:, 1], points[:, 0]) % (2 * pi)
    return angles


def convert_points_clockwise(points: array) -> array:
    mean_center = mean(points, axis=0)
    adjusted_points = points - mean_center

    angles = angle_between(adjusted_points)

    clockwise_points = points[argsort(angles)]

    return clockwise_points


def get_rect_points(x, y, width, height, mode='center', rotation=0):
    points = []

    if mode == 'center':
        points.append([x - width / 2, y - height / 2])
        points.append([x + width / 2, y - height / 2])
        points.append([x - width / 2, y + height / 2])
        points.append([x + width / 2, y + height / 2])

        center = array([x, y])
    else:
        points.append([x, y])
        points.append([x + width, y])
        points.append([x, y + height])
        points.append([x + width, y])

        center = array([x + width / 2, y + height / 2])

    points = convert_points_clockwise(array(points))

    points = rotate(points, center, rotation)

    return points


def rotate(p, origin=(0, 0), degrees=0):
    angle = degrees_to_radians(degrees)
    r = array([[cos(angle), -sin(angle)],
               [sin(angle),  cos(angle)]])
    o = atleast_2d(origin)
    p = atleast_2d(p)
    return squeeze((r @ (p.T-o.T) + o.T).T)


def get_polygon_centroid(points: array) -> array:
    """
    Find the polygon centroid based on polygon area.
    Note: this is different than taking the mean of
    all of the points. Assumes points are ordered in
    clockwise order.

    Args:
        points: N x 2 numpy array
    """

    points = convert_points_clockwise(points)

    first = points[0, :]
    last = points[-1, :]

    # If the first point doesn't match the last, add it back again
    if (first[0] !=  last[0]) or (first[1] or last [0]):
        points = append(points, first.reshape(1, 2), axis=0)

    twicearea = 0
    x = 0
    y = 0

    n_points = points.shape[0]

    i = 0
    j = n_points - 1
    while i < n_points:
        p1 = points[i, :]
        p2 = points[j, :]
        f = p1[0] * p2[1] - p2[0] * p1[1]
        twicearea += f
        x += (p1[0] + p2[0]) * f
        y += (p1[1] + p2[1]) * f

        j = i
        i += 1

    f = twicearea * 3
    return array([x / f, y / f])


def regular_polygon_points(x, y, radius, n_points, rot_angle=0):
    base_angle = pi / 2
    angle = pi * 2 / n_points
    points = empty((n_points, 2), float)
    points[:, 0] = x + cos(base_angle + rot_angle + arange(n_points) * angle) * radius
    points[:, 1] = y + sin(base_angle + rot_angle + arange(n_points) * angle) * radius
    return points


def round_to_pixels(points: array, pixel_size: float=1.0) -> array:
    return round(points * pixel_size) / pixel_size


def get_line(p1: array, p2: array, pixel_size=None) -> array:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    n = max(abs(dx), abs(dy))
    ninv = 0 if n == 0 else 1 / n

    x_step = dx * ninv
    y_step = dy * ninv

    steps = array([x_step, y_step])

    points = ones(shape=(n, 2)) * arange(n).reshape(n, 1) * steps + p1

    if pixel_size is not None:
        points = round_to_pixels(points, pixel_size)

    return points


def get_circle_fill_mask(arr_size: array, x: float, y: float, radius: float) -> array:
    # _x, _y = meshgrid(arange(arr_size[0]), arange(arr_size[1]))
    # _x, _y = _x.flatten(), _y.flatten()
    # points = vstack((_x, _y)).T
    #
    # dists = norm(points - (x, y),  axis=1)
    #
    # mask = dists <= radius
    # mask = mask.reshape(arr_size[0], arr_size[1])

    rr, cc = circle(x, y, radius, shape=arr_size)
    mask = zeros(arr_size, dtype=bool)
    mask[rr, cc] = True
    mask = mask[:, :, 0].reshape(mask.shape[0], mask.shape[1])

    return mask


def get_polygon_fill_mask(poly_points: array, arr_size: array) -> array:
    """
    :param poly_points: N x 2 array of X and Y coordinates of polygon points
    :param arr_size: X and Y dimensions of the grid to get the mask for
    :return:
    """

    # x, y = meshgrid(arange(arr_size[0]), arange(arr_size[1]))  # make a canvas with coordinates
    # x, y = x.flatten(), y.flatten()
    # points = vstack((x, y)).T
    #
    # p = Path(poly_points)
    # grid = p.contains_points(points)
    # mask = grid.reshape(arr_size[0], arr_size[1])

    mask = polygon2mask(arr_size, poly_points)
    mask = mask[:, :, 0].reshape(mask.shape[0], mask.shape[1])

    return mask


def create_edges_entry(points: array) -> List[dict]:
    # points = round_to_pixels(points, pixel_size)

    edges = []
    for i in range(points.shape[0]):
        edges.append(create_edge_entry(points[i, :], points[(i + 1) % points.shape[0], :]))

    return edges


def create_edge_entry(p1: array, p2: array) -> dict:
    edge = dict()

    edge['y_min'] = min(p1[1], p2[1])
    edge['y_max'] = max(p1[1], p2[1])

    if p1[0] < p2[0]:
        left = p1
        right = p2
    else:
        left = p2
        right = p1

    edge['x'] = left[0]

    if (right[0] - left[0]) == 0:
        edge['slope'] = None
    else:
        edge['slope'] = (right[1] - left[1]) / (right[0] - left[0])

    return edge


def get_bounding_box(points):
    min_x = points[0, :].min()
    max_x = points[0, :].max()
    min_y = points[1, :].min()
    max_y = points[1, :].max()

    return [min_x, max_x, min_y, max_y]


def circle_pack(x, y, max_radius, min_radius, radius_decrease, num_retries, buffer):
    circles = array([[random() * x, random() * y, max_radius]])

    curr_radius = max_radius

    while curr_radius >= min_radius:
        count_tries = 0
        while count_tries < num_retries:
            loc = array([random() * x, random() * y])
            count_overlaps = sum((norm(loc - circles[:, 0:2], axis=1) - circles[:, 2] - curr_radius - buffer) < 0)
            if count_overlaps == 0:
                circles = append(circles, array([[loc[0], loc[1], curr_radius]]), axis=0)
                count_tries = 0
            else:
                count_tries += 1
        curr_radius -= radius_decrease

    return circles


def polygon_centroid(points):
    # This assumes
    if points[0, :] != points[-1, :]:
        points = append(points, points[0, :])

    twice_area = 0
    x = 0
    y = 0
    num_points = points.shape[0]
    i = 0
    j = num_points - 1
    while i < num_points:
        p1 = points[i, :]
        p2 = points[j, :]
        f = p1[0] * p2[1] - p2[0] * p1[1]
        twice_area += f
        x += (p1[0] + p2[0]) * f
        y += (p1[1] + p2[1]) * f

    f = twice_area * 3
    return array([x / f, y / f])


def polygon_mean_center(points):
    return mean(points, axis=0)


def intersect_point(p1, p2, p3, p4):
    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) -
      (p4[1] - p3[1]) * (p1[0] - p3[0])) / \
      ((p4[1] - p3[1]) * (p2[0] - p1[0]) -
      (p4[0] - p3[0]) * (p2[1] - p1[1]))

    ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) -
      (p2[1] - p1[1]) * (p1[0] - p3[0])) / \
      ((p4[1] - p3[1]) * (p2[0] - p1[0]) -
      (p4[0] - p3[0]) * (p2[1] - p1[1]))

    x = p1[0] + ua * (p2[0] - p1[0])
    y = p1[1] + ua * (p2[1] - p1[1])

    intersection = array([x, y])

    if (ua < 0) | (ua > 1) | (ub < 0) | (ub > 1):
        intersection = None

    return intersection


def get_lines_from_points(points):
    lines = []
    for i in range(points.shape[0]):
        lines.append([points[i, :], points[(i + 1) % points.shape[0], :]])
    return lines


def get_points_from_lines(lines):
    points_list = []
    for line in lines:
        points_list.append(line[0])
    return array(points_list)


def convert_points_clockwise2(points):
    mean_center = polygon_mean_center(points)
    point_lines = points - mean_center
    point_lines_unit = point_lines / norm(point_lines)
    reference_line = array([0, 1])
    dot_product = clip(dot(point_lines_unit, reference_line), -1.0, 1.0)
    angles = arccos(dot_product)
    return points[argsort(angles)]


class CuttablePolygon(object):
    def __init__(self, points, times_split=0):
        self.points = convert_points_clockwise(points)
        self.center = self.get_center()
        self.times_split = times_split

    def get_center(self):
        centroid = get_polygon_centroid(self.points)
        return centroid

    def get_shrunken_points(self, ratio):
        return self.points * ratio + self.center * (1 - ratio)

    def intersect_points(self, p1, p2):
        # TODO: fix this, assigning one point to opposite poly somewhere
        intersections = []
        intersect_line_inds = []
        lines = get_lines_from_points(self.points)
        for i, line in enumerate(lines):
            intersection = intersect_point(p1, p2, line[0], line[1])
            if intersection is not None:
                intersections.append(intersection)
                intersect_line_inds.append(i)

        if len(intersections) == 2:

            if intersect_line_inds[0] > intersect_line_inds[1]:
                intersections = [intersections[1], intersections[0]]
                intersect_line_inds = [intersect_line_inds[1], intersect_line_inds[0]]

            return intersections, intersect_line_inds
        else:
            return None, None

    def split(self, p1, p2):

        lines = get_lines_from_points(self.points)

        intersect_points, intersect_line_inds = self.intersect_points(p1, p2)

        if intersect_points is None:
            return [self]

        poly_lines_1 = []
        poly_lines_2 = []

        num_lines = len(lines)
        checkpoints = 0

        for i in range(num_lines):
            if checkpoints == 0:
                if i in intersect_line_inds:
                    poly_lines_1.append([intersect_points[1], intersect_points[0]])
                    poly_lines_1.append([intersect_points[0], lines[i][1]])
                    poly_lines_2.append([lines[i][0], intersect_points[0]])
                    poly_lines_2.append([intersect_points[0], intersect_points[1]])
                    checkpoints += 1
                else:
                    poly_lines_2.append(lines[i])
            elif checkpoints == 1:
                if i in intersect_line_inds:
                    poly_lines_1.append([lines[i][0], intersect_points[1]])
                    poly_lines_2.append([intersect_points[1], lines[i][1]])
                    checkpoints += 1
                else:
                    poly_lines_1.append(lines[i])
            else:
                poly_lines_2.append(lines[i])

        poly_pts_1 = get_points_from_lines(poly_lines_1)
        poly_pts_2 = get_points_from_lines(poly_lines_2)

        if poly_pts_1.shape[0] > 0 and poly_pts_2.shape[0] > 0:

            poly_1 = CuttablePolygon(poly_pts_1, self.times_split + 1)
            poly_2 = CuttablePolygon(poly_pts_2, self.times_split + 1)

            return [poly_1, poly_2]

        return [self]


def split_polygons(polygons, cut_lines):

    for line in cut_lines:
        new_polys = []
        for poly in polygons:
            split_polys = poly.split(line[0], line[1])
            for sp in split_polys:
                new_polys.append(sp)

        polygons = new_polys

    return polygons
