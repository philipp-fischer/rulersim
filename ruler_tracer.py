import numpy as np
import matplotlib.pyplot as plt
import sys
import math

import collada
from collada.lineset import BoundLineSet, Line, LineSet
from collada.scene import Scene, Node, NodeNode
from collada.geometry import BoundGeometry, Geometry
from lxml.etree import _Element
from ruler_tracer_tools import *

from pypcd import pypcd

def index_nodes(nodes, name_dict, level=0, name=''):
    for n in nodes:
        own_name = n.xmlnode.get('name')
        print('.'*level, n, name, n.xmlnode.get('id'))

        if 'node' in dir(n):
            id = n.node.id
        elif 'geometry' in dir(n):
            id = n.geometry.id
        else:
            id = n.xmlnode.get('id')

        if id is not None:
            if own_name is not None:
                name_dict[id] = own_name
            else:
                name_dict[id] = name

        if isinstance(n, Node):
            if own_name is not None:
                index_nodes(n.children, name_dict, level + 1, name=own_name)
            else:
                index_nodes(n.children, name_dict, level + 1, name=name)


def create_scene(filename):
    scene = {}
    scene['triangles'] = []
    scene['rulers'] = {}

    dae = collada.Collada(filename, ignore=[collada.DaeUnsupportedError,
                                            collada.DaeBrokenRefError])

    assert (isinstance(dae.scene, Scene))
    name_dict = {}
    index_nodes(dae.scene.nodes, name_dict)
    print(name_dict)

    for geom in dae.scene.objects('geometry'):
        print('===================')
        assert (isinstance(geom, BoundGeometry))
        assert (isinstance(geom.original, Geometry))
        assert (isinstance(geom.original.xmlnode, _Element))

        geometry_name = name_dict[geom.original.id]
        # print('>', geom, 'NAME:', name_dict[geom.original.id])

        for prim in geom.primitives():
            print('>', prim)
            prim_type = type(prim).__name__
            triangles = None
            lines = None
            if prim_type == 'BoundTriangleSet':
                triangles = prim
            elif prim_type == 'BoundPolylist':
                triangles = prim.triangleset()
            elif prim_type == 'BoundLineSet':
                assert (isinstance(prim, BoundLineSet))

                lines = prim.lines()
            else:
                print('Unsupported mesh used:', prim_type)

            print(type(prim))

            if triangles is not None and 'Ruler' not in geometry_name:
                print('=== Triangles')
                vertices = triangles.vertex.flatten().tolist()
                batch_len = len(vertices) // 3
                indices = triangles.vertex_index.flatten().tolist()
                normals = triangles.normal.flatten().tolist()
                print(vertices)
                print(indices)
                print(normals)

                i = np.array(indices)
                v = np.array(vertices).reshape((-1, 3))
                points = v[i].reshape((-1, 3, 3)) / 20

                for tidx in range(points.shape[0]):
                    print("Adding ", points[tidx])
                    scene['triangles'].append(add_triangle(points[tidx], [1, .3, .1]))

            if lines is not None:
                print('=== Lines')
                if 'Ruler' in geometry_name:
                    assert('_' in geometry_name)
                    ruler_name, part = geometry_name.split('_', 2)

                    first_line = list(lines)[0]
                    assert (isinstance(first_line, Line))
                    i = np.array(first_line.indices)
                    v = np.array(first_line.vertices).reshape((-1, 3))
                    points = v[i].reshape((-1, 2, 3)) / 20

                    print(first_line)
                    print(first_line.vertices)
                    print(first_line.indices)

                    if ruler_name not in scene['rulers']:
                        scene['rulers'][ruler_name] = {}

                    scene['rulers'][ruler_name][part] = points

    return scene

def project_laser_points(scene, LaserO, LaserD, LaserLeft):
    # TODO: Go over angles, project rays into scene, store points
    angles_range = (-35, 35)
    num_steps = 100
    laser_points = []
    for idx in range(num_steps):
        angle = (angles_range[1] - angles_range[0]) / (num_steps-1) * idx + angles_range[0]
        #print(angle)
        vec = LaserLeft * math.sin(math.radians(angle)) + LaserD * math.cos(math.radians(angle))

        # Trace this laser ray
        traced = trace_ray(scene['triangles'], LaserO, vec, LaserO, LaserO)  # TODO: remove lighting stuff
        if not traced:
            continue
        obj, M, N, col_ray = traced
        laser_points.append(M)

    if len(laser_points) > 0:
        return np.vstack(laser_points)
    else:
        return None




def simulate_ruler_single(scene, ruler, x_offset):
    offset_vector = np.array([x_offset, 0, 0])

    line_laser = ruler['Laser'][0]

    LaserO = line_laser[0] + offset_vector
    LaserD = normalize(line_laser[1] - line_laser[0])

    line_cam = ruler['Cam'][0]
    CamO = line_cam[0] + offset_vector
    CamD = normalize(line_cam[1] - line_cam[0])
    LaserLeft = normalize(np.cross(LaserD, CamO - LaserO))

    laser_points = project_laser_points(scene, LaserO, LaserD, LaserLeft)
    # print(laser_points)

    if laser_points is not None:

        pcd_points = np.zeros((len(laser_points), 4), dtype=np.float32)
        pcd_points[:, 0:3] = laser_points

        colors = np.zeros((len(laser_points), 3), dtype=np.uint)

        for idx, laser_point in enumerate(laser_points):
            # Check visibility from camera (like shadow checking in ray tracer)
            toCam = CamO - laser_point
            l = []
            for obj_sh in scene['triangles']:
                dist = intersect(laser_point, toCam, obj_sh)
                if dist > 0.001:
                    l.append(dist)

            if l and min(l) < np.inf:
                colors[idx, :] = [255, 0, 0]
            else:
                colors[idx, :] = [0, 0, 255]

        colors_uint = np.array((colors[:, 0] << 16) | (colors[:, 1] << 8) | (colors[:, 2] << 0), dtype=np.uint32)
        colors_uint.dtype = np.float32
        pcd_points[:, 3] = colors_uint

        print("Ret points with offset ", x_offset)
        return pcd_points
    else:
        return None

def simulate_ruler(scene, ruler):
    print("Simulating Ruler '%s'" % str(ruler))

    all_points = []
    for x_offset in np.linspace(-0.3, 0.3, 200):
        scanline_points = simulate_ruler_single(scene, ruler, x_offset)
        if scanline_points is not None:
            all_points.append(scanline_points)

    all_pcd_points = np.vstack(all_points)

    result = pypcd.make_xyz_rgb_point_cloud(all_pcd_points)
    pypcd.save_point_cloud_bin(result, 'out.pcd')


if __name__ == '__main__':

    w = 200
    h = 150

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


    scene = create_scene('data/rulertest02.dae')
    print(scene)

    for ruler in scene['rulers']:
        simulate_ruler(scene, scene['rulers'][ruler])

    sys.exit(1)

    depth_max = 2  # Maximum number of light reflections.
    col = np.zeros(3)  # Current color.
    O = np.array([0., -1, 0.45])  # Camera.
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
            Q[0] = x
            Q[2] = y
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
