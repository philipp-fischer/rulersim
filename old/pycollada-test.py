import collada
from collada.lineset import BoundLineSet, Line, LineSet
from collada.scene import Scene, Node, NodeNode
from collada.geometry import BoundGeometry, Geometry
from lxml.etree import _Element


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

if __name__ == '__main__':
    filename = r'data/rulertest01.dae'
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
        # ElementTree

        # print('>', geom, 'NAME:', geom.original.xmlnode.getparent())
        print('>', geom, 'NAME:', name_dict[geom.original.id])

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

                # print(prim.original.name)
                lines = prim.lines()
            else:
                print('Unsupported mesh used:', prim_type)

            print(type(prim))

            if triangles is not None:
                print('=== Triangles')
                vertices = triangles.vertex.flatten().tolist()
                batch_len = len(vertices) // 3
                indices = triangles.vertex_index.flatten().tolist()
                normals = triangles.normal.flatten().tolist()
                print(vertices)
                print(indices)
                print(normals)

            if lines is not None:
                print('=== Lines')

                first_line = list(lines)[0]
                print(first_line)
                assert (isinstance(first_line, Line))
                print(first_line.vertices)
                print(first_line.indices)
