import numpy as np
from scipy.spatial import distance


def compute_distances(coords):
    D = distance.cdist(coords, coords)
    idx_below = np.tril_indices_from(D, k=-1)
    d = D[idx_below]
    return d


def compute_variogram(points, displacements):
    h = compute_distances(points)
    dz = compute_distances(displacements)
    return h, dz



if __name__ == "__main__":
    coords = [(35.0456, -85.2672),
              (35.1174, -89.9711),
              (35.9728, -83.9422),
              (36.1667, -86.7833)]
    h = compute_distances(coords)
    print(h)
