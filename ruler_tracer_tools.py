import numpy as np
import matplotlib.pyplot as plt

settings = {}

def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_triangle(O, D, triangle):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # triangle, or +inf if there is no intersection.
    # O and P are 3D points
    points = triangle['points']

    # Triangle edges:
    U = points[1] - points[0]
    V = points[2] - points[0]

    N = triangle['normal']

    # Intersect with plane
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(points[0] - O, N) / denom
    if d < 0:
        return np.inf

    # Point of intersection
    PI = O + D*d

    # Get coords in triangle
    W = PI - points[0]

    UU = np.dot(U, U)
    UV = np.dot(U, V)
    VV = np.dot(V, V)
    UV2 = UV ** 2

    WV = np.dot(W, V)
    WU = np.dot(W, U)

    s = (UV * WV - VV * WU) / (UV2 - UU * VV)
    t = (UV * WU - UU * WV) / (UV2 - UU * VV)

    if s < 0 or t < 0 or s+t > 1:
        return np.inf
    else:
        return d


def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj)
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])


def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


def trace_ray(scene, rayO, rayD, O, L):
    global settings
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Shadow: find if the point is shadowed or not.
    l = [intersect(M + N * .0001, toL, obj_sh)
         for k, obj_sh in enumerate(scene) if k != obj_idx]
    if l and min(l) < np.inf:
        return
    # Start computing the color.
    col_ray = settings['ambient']
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', settings['diffuse_c']) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', settings['specular_c']) * max(np.dot(N, normalize(toL + toO)), 0) ** settings['specular_k'] * settings['color_light']
    return obj, M, N, col_ray


def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color), reflection=.5)


def add_plane(position, normal):
    global settings
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (settings['color_plane0']
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else settings['color_plane1']),
                diffuse_c=.75, specular_c=.5, reflection=.25)


def add_triangle(points, color):
    nppoints = np.array(points).astype(np.float32)
    normal = normalize(np.cross(nppoints[1]-nppoints[0], nppoints[2]-nppoints[0]))
    return dict(type='triangle', points=nppoints, normal=normal,
                color=np.array(color))


