# TODO: circle rasterization
# TODO: arbitrary polygon rasterization
# TODO: line intersection
# TODO: point in triangle
# TODO: voronoi/delaunay triangulation??
# TODO: marching squares


import numpy as np


def chiakins_curve(coords, refinements=5):
    # TODO: modify to retain original points? closed version
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords
