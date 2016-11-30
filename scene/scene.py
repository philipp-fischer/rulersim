import numpy as np
import collada
from collada.lineset import BoundLineSet, Line, LineSet
from collada.scene import Scene as ColladaScene
from collada.scene import Node, NodeNode
from collada.geometry import BoundGeometry, Geometry
from lxml.etree import _Element

from scene.triangle import Triangle
from scene.primitive import Primitive
from scene.ruler import Ruler


class Scene:
    def __init__(self):
        self.primitives = []
        self.rulers = {}

    def add_primitive(self, object):
        assert(isinstance(object, Primitive))

        self.primitives.append(object)

    def get_primitives(self):
        return self.primitives

    @staticmethod
    def _index_dae_nodes(nodes, name_dict, level=0, name=''):
        for n in nodes:
            own_name = n.xmlnode.get('name')
            print('.' * level, n, name, n.xmlnode.get('id'))

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
                    Scene._index_dae_nodes(n.children, name_dict, level + 1, name=own_name)
                else:
                    Scene._index_dae_nodes(n.children, name_dict, level + 1, name=name)

    def load_from_dae(self, filename):
        dae = collada.Collada(filename, ignore=[collada.DaeUnsupportedError,
                                                collada.DaeBrokenRefError])

        global_scale = 25.4

        assert (isinstance(dae.scene, ColladaScene))
        name_dict = {}
        self._index_dae_nodes(dae.scene.nodes, name_dict)
        print(name_dict)

        ruler_dict = {}

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
                    points = v[i].reshape((-1, 3, 3)) * global_scale

                    for tidx in range(points.shape[0]):
                        print("Adding ", points[tidx])
                        self.add_primitive(Triangle(points[tidx]))

                if lines is not None:
                    print('=== Lines')
                    if 'Ruler' in geometry_name:
                        assert ('_' in geometry_name)
                        ruler_name, part = geometry_name.split('_', 2)

                        first_line = list(lines)[0]
                        assert (isinstance(first_line, Line))
                        i = np.array(first_line.indices)
                        v = np.array(first_line.vertices).reshape((-1, 3))
                        points = v[i].reshape((-1, 2, 3)) * global_scale

                        print(first_line)
                        print(first_line.vertices)
                        print(first_line.indices)

                        if ruler_name not in ruler_dict:
                            ruler_dict[ruler_name] = {}

                        ruler_dict[ruler_name][part] = points

        for ruler_name in ruler_dict:
            self.rulers[ruler_name] = Ruler(ruler_dict[ruler_name])

    def trace_rays(self, rays, exclude_primitives=None):
        # Initialize with list of invalid points at infinity
        shortest_dists = np.full((rays.shape[0]), np.inf)
        closest_int_points = np.full((rays.shape[0], 3), np.nan)
        intersection_primitives = np.zeros(shape=(rays.shape[0],), dtype=np.int32)

        # Now check for closest intersections with the scene primitives
        for idx, primitive in enumerate(self.primitives):
            distances, intersection_points = primitive.intersect_rays(rays)
            mask = (distances < shortest_dists)

            if exclude_primitives is not None:
                mask[exclude_primitives == idx] = False

            shortest_dists[mask] = distances[mask]
            closest_int_points[mask, :] = intersection_points[mask, :]
            intersection_primitives[mask] = idx

        return shortest_dists, closest_int_points, intersection_primitives