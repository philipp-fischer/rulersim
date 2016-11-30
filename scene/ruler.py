import numpy as np
from scene.primitive import normalize
import math
import PIL.Image as Image
import time


class Ruler:
    """
    Simulates a ruler device.
    Specs about the Ruler E1200:
    - Resolution 1024x512
    - Camera opening angles HxV = approx. 31 deg x 62 deg
    - Basline between camera and laser origin = 317.2mm (z offset: laser is 15.2mm higher)
    - Angle between camera center line and laser sheet = 31 deg


    """
    def __init__(self, components):
        self.components = components
        self.laser_angle_range = (-40, 40)
        self.camera_horizontal_angle_range = (-31, 31)
        self.camera_vertical_angle_range = (-31/2, 31/2)  # TODO: is this correct?
        self.camera_resolution = (1024, 512)

        self.num_laser_steps = 1000  # sampling along single laser line

    def project_laser_points(self, scene, origin, direction, left, return_only_valid=False):
        from scene.scene import Scene
        assert(isinstance(scene, Scene))

        laser_rays = np.zeros((self.num_laser_steps, 2, 3))  # (RAYS, OriginOrDir, XYZ)
        left = normalize(left)
        direction = normalize(direction)

        # Create array of laser rays
        for idx in range(self.num_laser_steps):
            angle = (self.laser_angle_range[1] - self.laser_angle_range[0]) / (self.num_laser_steps - 1) * idx \
                    + self.laser_angle_range[0]
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

    def create_camera_image(self, h_angles, v_angles, suffix=''):
        img = np.zeros(self.camera_resolution[::-1], dtype=np.uint8)
        num_pts = h_angles.shape[0]
        x_coords = np.round((h_angles - self.camera_horizontal_angle_range[0]) /
                   (self.camera_horizontal_angle_range[1] - self.camera_horizontal_angle_range[0]) *
                   (self.camera_resolution[0]-1)).astype(np.int32)
        y_coords = np.round((v_angles - self.camera_vertical_angle_range[0]) /
                   (self.camera_vertical_angle_range[1] - self.camera_vertical_angle_range[0]) *
                   (self.camera_resolution[1]-1)).astype(np.int32)

        for idx in range(num_pts):
            if x_coords[idx] > img.shape[1] or y_coords[idx] > img.shape[0]:
                print(y_coords[idx], x_coords[idx])
            else:
                img[y_coords[idx], x_coords[idx]] = 255

        Image.fromarray(img).save('camera%s.png' % suffix)

    def compute_out_of_view_points(self, intersection_points, CamO, CamLeft, CamUp):
        # Laser scene points relative to camera origin
        trans_points = intersection_points - CamO
        # Normalize these vectors to length 1
        trans_points /= np.linalg.norm(trans_points, axis=1).reshape((-1, 1))

        v_angles = np.rad2deg(np.arccos(np.matmul(trans_points, CamUp))) - 90
        h_angles = np.rad2deg(np.arccos(np.matmul(trans_points, CamLeft))) - 90

        mask = (h_angles >= self.camera_horizontal_angle_range[0]) & \
               (h_angles <= self.camera_horizontal_angle_range[1]) & \
               (v_angles >= self.camera_vertical_angle_range[0]) & \
               (v_angles <= self.camera_vertical_angle_range[1])

        self.create_camera_image(h_angles[mask], v_angles[mask], suffix="%08d" % int(time.clock()*1000))

        return ~mask

    def simulate_single_scan(self, scene, x_offset):
        offset_vector = np.array([x_offset, 0, 0])

        line_laser = self.components['Laser'][0]  # TODO: Can be multiple lasers

        LaserO = line_laser[0] + offset_vector
        LaserD = normalize(line_laser[1] - line_laser[0])

        line_cam = self.components['Cam'][0]
        CamO = line_cam[0] + offset_vector
        CamD = normalize(line_cam[1] - line_cam[0])

        LaserLeft = normalize(np.cross(LaserD, CamO - LaserO))
        CamUp = normalize(np.cross(CamD, LaserLeft))

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

            colors[:, :] = [0, 0, 255]

            # TODO:
            # First compute h_angles and v_angles in camera
            # Then do the occlusion and visibility things
            # Then write out a colored (and named) camera image using the computed flags

            # Mark occluded points as not visible
            occluded_mask = np.isfinite(distances)
            colors[occluded_mask, :] = [255, 0, 0]

            # Mark out-of-view points (not in camera field of view)
            out_of_view_mask = self.compute_out_of_view_points(laser_points, CamO, LaserLeft, CamUp)
            colors[out_of_view_mask, :] = [200, 50, 200]

            colors_uint = np.array((colors[:, 0] << 16) | (colors[:, 1] << 8) | (colors[:, 2] << 0), dtype=np.uint32)
            colors_uint.dtype = np.float32
            pcd_points[:, 3] = colors_uint

            return pcd_points
        else:
            return None