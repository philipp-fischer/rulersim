import abc
import numpy as np


def normalize(x):
    x /= np.linalg.norm(x)
    return x


class Primitive:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def intersect_rays(self, rays):
        """This methods intersects a 3D array of rays (RAYS, OriginOrDir, XYZ) with this primitve.
           It returns the distances and points of intersection"""
        return np.full((rays.shape[0]), np.inf), np.full((rays.shape[0], 3), np.nan)

    def get_color(self, position):
        """Returns the object color at the given position as RGB tuple"""
        return 0, 0, 0

