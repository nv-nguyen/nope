import math
import os
import random
import sys
import time
from typing import Tuple
import bpy
from mathutils import Vector, Matrix
import numpy as np


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


# Blender: camera looks in negative z-direction, y points up, x points right.
# Opencv: camera looks in positive z-direction, y points down, x points right.
def cv_cam2world_to_bcam2world(cv_cam2world):
    """
    :cv_cam2world: numpy array.
    :return:
    """
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    cam_location = Vector(cv_cam2world[:3, -1].tolist())
    cv_cam2world_rot = Matrix(cv_cam2world[:3, :3].tolist())

    cv_world2cam_rot = cv_cam2world_rot.transposed()
    cv_translation = -1.0 * cv_world2cam_rot @ cam_location

    blender_world2cam_rot = R_bcam2cv @ cv_world2cam_rot
    blender_translation = R_bcam2cv @ cv_translation

    blender_cam2world_rot = blender_world2cam_rot.transposed()
    blender_cam_location = -1.0 * blender_cam2world_rot @ blender_translation

    blender_matrix_world = Matrix(
        (
            blender_cam2world_rot[0][:] + (blender_cam_location[0],),
            blender_cam2world_rot[1][:] + (blender_cam_location[1],),
            blender_cam2world_rot[2][:] + (blender_cam_location[2],),
            (0, 0, 0, 1),
        )
    )

    return blender_matrix_world


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(scale_scene=1.0):
    bbox_min, bbox_max = scene_bbox()
    scale = scale_scene / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def add_lights(name, location, euler0):
    if name != "light":
        bpy.ops.object.light_add(type="POINT")
    lamp = bpy.data.lights[name]
    lamp.use_shadow = False
    lamp.specular_factor = 0.0
    lamp.energy = 100.0
    bpy.data.objects[name].location = location
    bpy.data.objects[name].scale[0] = 100
    bpy.data.objects[name].scale[1] = 100
    bpy.data.objects[name].scale[2] = 100
    bpy.data.objects[name].rotation_euler[0] = euler0


def set_camera_focal_length_in_world_units(camera_data, focal_length):
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera_data.sensor_width
    sensor_height_in_mm = camera_data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camera_data.sensor_fit == "VERTICAL":
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    camera_data.lens = focal_length / s_u


class BlenderInterface:
    def __init__(self, gpu_ids, name_engine, resolution):
        self.resolution = resolution

        # Delete the default cube (default selected)
        bpy.ops.object.delete()

        # Deselect all. All new object added to the scene will automatically selected.
        assert name_engine in ["CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"]
        self.scene = bpy.context.scene
        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.engine = name_engine
        self.blender_renderer.image_settings.file_format = "PNG"
        self.blender_renderer.image_settings.color_mode = "RGBA"
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100

        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        prefs = bpy.context.preferences
        bpy.context.scene.render.engine = "CYCLES"
        cprefs = prefs.addons["cycles"].preferences
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        bpy.context.scene.cycles.device = "GPU"
        cprefs.compute_device_type = "CUDA"

        self.scene.cycles.device = "GPU"
        self.scene.cycles.samples = 32
        self.scene.cycles.diffuse_bounces = 1
        self.scene.cycles.glossy_bounces = 1
        self.scene.cycles.transparent_max_bounces = 3
        self.scene.cycles.transmission_bounces = 3
        self.scene.cycles.filter_width = 0.01
        self.scene.cycles.use_denoising = True
        self.scene.render.film_transparent = True

        position = 1
        idx_light = 0
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [0, 1]:
                    if idx_light == 0:
                        name_light = "Light"
                    elif idx_light == 1:
                        name_light = f"Point"
                    else:
                        name_light = f"Point.{idx_light:03d}"
                    add_lights(
                        name_light,
                        (position * x, position * y, position * z),
                        euler0=0,
                    )
                    idx_light += 1

        # Set up the camera
        self.camera = bpy.context.scene.camera
        self.camera.data.sensor_height = self.camera.data.sensor_width  # Square sensor
        set_camera_focal_length_in_world_units(
            self.camera.data, 525.0 / 512 * resolution
        )  # Set focal length to a common value (kinect)
        bpy.ops.object.select_all(action="DESELECT")

    def import_mesh(self, fpath, scale=1.0, object_world_matrix=None):
        bpy.ops.import_scene.gltf(filepath=str(fpath), merge_vertices=True)
        obj = bpy.context.selected_objects[0]

        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.rotation_mode = "XYZ"
        obj.location = (0.0, 0.0, 0.0)  # center the bounding box!
        obj.rotation_euler[0] = -math.pi / 2
        if scale != 1.0:
            bpy.ops.transform.resize(value=(scale, scale, scale))
        scene_scale, scene_offset = normalize_scene(scale_scene=0.8)
        return scene_scale, scene_offset

    def render(self, output_dir, blender_cam2world_matrices, write_cam_params=False):
        for i in range(len(blender_cam2world_matrices)):
            self.camera.matrix_world = blender_cam2world_matrices[i]
            img_path = os.path.join(output_dir, "%06d.png" % i)

            # Render the color image
            self.blender_renderer.filepath = img_path
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cad_path",
        type=str,
    )
    parser.add_argument("--pose_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument(
        "--disable_output", action="store_true", help="Disable output of blender"
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    if args.disable_output == "true":
        # redirect output to log file
        logfile = args.save_path + "/render.log"
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

    obj_poses = np.load(args.pose_path)
    cam_poses = [inverse_transform(obj_pose) for obj_pose in obj_poses]
    blender_poses = [cv_cam2world_to_bcam2world(m) for m in cam_poses]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    renderer = BlenderInterface(args.gpu_id, "CYCLES", resolution=512)
    renderer.import_mesh(args.cad_path, scale=1.0, object_world_matrix=np.eye(4))
    renderer.render(args.output_dir, blender_poses, write_cam_params=False)
    
    if args.disable_output == "true":
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
