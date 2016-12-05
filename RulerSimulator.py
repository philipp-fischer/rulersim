from scene.scene import *
from scene.triangle import *
from pypcd import pypcd

if __name__ == '__main__':
    scene = Scene()

    scene.load_from_dae(r'data/rulertest04.dae')

    for ruler_name in scene.rulers:
        all_points = []
        for x_offset in np.linspace(-200, 200, 100):
            scanline_points = scene.rulers[ruler_name].simulate_single_scan(scene, x_offset, ruler_name)
            if scanline_points is not None:
                all_points.append(scanline_points)

        all_pcd_points = np.vstack(all_points)

        result = pypcd.make_xyz_rgb_point_cloud(all_pcd_points)
        pypcd.save_point_cloud_bin(result, 'out_%s.pcd' % ruler_name)
