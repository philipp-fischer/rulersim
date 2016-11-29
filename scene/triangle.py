from scene.primitive import *
import numpy as np


class Triangle(Primitive):
    def __init__(self, points):
        assert(len(points) == 3)

        self.points = np.array(points).astype(np.float32)
        self.N = normalize(np.cross(self.points[1] - self.points[0], self.points[2] - self.points[0]))

        self.U = points[1] - points[0]
        self.V = points[2] - points[0]

        self.UU = np.dot(self.U, self.U)
        self.UV = np.dot(self.U, self.V)
        self.VV = np.dot(self.V, self.V)
        self.UV2 = self.UV ** 2

        self.UV2UUVV = self.UV2 - self.UU * self.VV

    def intersect_rays(self, rays):
        assert(rays.shape[1:] == (2, 3))  # Should be [N, 2, 3]

        O = rays[:, 0, :]
        D = rays[:, 1, :]

        distances = np.zeros((rays.shape[0]), dtype=np.float32)

        # Intersect with plane
        denom = np.matmul(D, self.N)

        np.seterr(divide='ignore')
        distances = np.matmul(self.points[0] - O, self.N) / denom
        np.seterr(divide='warn')

        distances[np.abs(denom) < 1e-6] = np.inf  # Not intersecting
        distances[distances < 0] = np.inf  # Wrong side of triangle (normal)

        # Points of intersections
        np.seterr(invalid='ignore')
        PI = O + D * distances.reshape((-1, 1))
        np.seterr(invalid='warn')


        # Get coords in triangle
        W = PI - self.points[0]

        WV = np.matmul(W, self.V)
        WU = np.matmul(W, self.U)

        s = (self.UV * WV - self.VV * WU) / self.UV2UUVV
        t = (self.UV * WU - self.UU * WV) / self.UV2UUVV

        np.seterr(invalid='ignore')
        distances[~np.isfinite(s) | ~np.isfinite(t) | (s < 0) | (t < 0) | (s+t > 1)] = np.inf
        np.seterr(invalid='warn')

        return distances, PI


def test_triangle():
    t = Triangle(
        np.array([[0,0,0],[10,0,0],[0,10,0]])
    )

    rays = np.array([[[0.1, 0.1, -2], [0, 0, 1]],
                     [[0.1, 0.1, -2], [0, 0, -1]],
                     [[0.5, 0.5, -1], [1, 0, 0]],
                     [[0, 0.2, -1], [1, 0, 1]]])
    print(rays.shape)

    dists, pts = t.intersect_rays(rays)
    print(dists)
    print(pts)

