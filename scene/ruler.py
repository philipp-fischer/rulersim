import numpy as np
from scene.primitive import normalize
import math


class Ruler:
    def __init__(self, components):
        self.components = components
        self.angle_range = (-35, 35)
        self.num_laser_steps = 100  # sampling along single laser line

    def project_laser_points(self, scene, origin, direction, left, return_only_valid=False):
        from scene.scene import Scene
        assert(isinstance(scene, Scene))

        laser_rays = np.zeros((self.num_laser_steps, 2, 3))  # (RAYS, OriginOrDir, XYZ)
        left = normalize(left)
        direction = normalize(direction)

        # Create array of laser rays
        for idx in range(self.num_laser_steps):
            angle = (self.angle_range[1] - self.angle_range[0]) / (self.num_laser_steps - 1) * idx + self.angle_range[0]
            # print(angle)
            vec = left * math.sin(math.radians(angle)) + direction * math.cos(math.radians(angle))

            laser_rays[idx, 0, :] = origin
            laser_rays[idx, 1, :] = vec

        # Trace rays into scene
        distances, intersection_points, int_primitives = scene.trace_rays(laser_rays)
        if return_only_valid:
            return intersection_points[np.isfinite(distances)], int_primitives[np.isfinite(distances)]
        else:
            return intersection_points, int_primitives

    def simulate_single_scan(self, scene, x_offset):
        offset_vector = np.array([x_offset, 0, 0])

        line_laser = self.components['Laser'][0]  # TODO: Can be multiple lasers

        LaserO = line_laser[0] + offset_vector
        LaserD = normalize(line_laser[1] - line_laser[0])

        line_cam = self.components['Cam'][0]
        CamO = line_cam[0] + offset_vector
        CamD = normalize(line_cam[1] - line_cam[0])
        LaserLeft = normalize(np.cross(LaserD, CamO - LaserO))

        # Project laser into scene and get intersection points (and corresponding primitives of intersection)
        laser_points, laser_primitives = self.project_laser_points(scene, LaserO, LaserD, LaserLeft, return_only_valid=True)

        # Now trace from these points in the scene to the ruler camera to check if they are visible
        if laser_points.shape[0] > 0:

            pcd_points = np.zeros((len(laser_points), 4), dtype=np.float32)
            pcd_points[:, 0:3] = laser_points

            directions_to_cam = CamO - laser_points

            scene_to_cam_rays = np.zeros((laser_points.shape[0], 2, 3))  # (RAYS, OriginOrDir, XYZ)
            scene_to_cam_rays[:, 0, :] = laser_points
            scene_to_cam_rays[:, 1, :] = directions_to_cam

            # Trace rays from surface point to camera and look for occlusion
            # (do not intersect with the primitive that the ray is traced from)
            distances, intersection_points, _ = scene.trace_rays(scene_to_cam_rays, exclude_primitives=laser_primitives)

            colors = np.zeros((len(laser_points), 3), dtype=np.uint)

            visible_mask = ~np.isfinite(distances)

            colors[visible_mask, :] = [0, 0, 255]
            colors[~visible_mask, :] = [255, 0, 0]

            colors_uint = np.array((colors[:, 0] << 16) | (colors[:, 1] << 8) | (colors[:, 2] << 0), dtype=np.uint32)
            colors_uint.dtype = np.float32
            pcd_points[:, 3] = colors_uint

            return pcd_points
        else:
            return None