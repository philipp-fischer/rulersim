import numpy as np
import matplotlib.pyplot as plt
from ruler_tracer_tools import *

if __name__ == '__main__':

    w = 400
    h = 300

    # List of objects.
    settings['color_plane0'] = 1. * np.ones(3)
    settings['color_plane1'] = 0. * np.ones(3)

    # Light position and color.
    L = np.array([5., 5., -10.])
    settings['color_light'] = np.ones(3)

    # Default light and material parameters.
    settings['ambient'] = .05
    settings['diffuse_c'] = 1.
    settings['specular_c'] = 1.
    settings['specular_k'] = 50

    scene = [add_sphere([0, .1, 2.3], .6, [0., 0., 1.]),
             # add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
             add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
             add_plane([0., -.5, 0.], [0., 1., 0.]),
             add_triangle([[0, -.5, 2], [-1, -.5, 2], [-.5, 0.5, 2]], [1, .3, .1])
             # add_plane([0., -.5, 2.], [0., 0., -1.]),
             ]

    depth_max = 5  # Maximum number of light reflections.
    col = np.zeros(3)  # Current color.
    O = np.array([0., 0.35, -1.])  # Camera.
    Q = np.array([0., 0., 0.])  # Camera pointing to.
    img = np.zeros((h, w, 3))

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1., -1. / r + .25, 1., 1. / r + .25)

    # Loop through all pixels.
    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        if i % 10 == 0:
            print(i / float(w) * 100, "%")
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            col[:] = 0
            Q[:2] = (x, y)
            D = normalize(Q - O)
            depth = 0
            rayO, rayD = O, D
            reflection = 1.
            # Loop through initial and secondary rays.
            while depth < depth_max:
                traced = trace_ray(scene, rayO, rayD, O, L)
                if not traced:
                    break
                obj, M, N, col_ray = traced
                # Reflection: create a new ray.
                rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
                depth += 1
                col += reflection * col_ray
                reflection *= obj.get('reflection', 1.)
            img[h - j - 1, i, :] = np.clip(col, 0, 1)

    plt.imsave('fig2.png', img)
