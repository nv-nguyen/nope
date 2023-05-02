import blenderproc as bproc
import numpy as np
import argparse
import os, sys
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def render_blender_proc(
    cad_path,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    tless_like,
):

    bproc.init()
    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_K_matrix(intrinsic, img_size[1], img_size[0])
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([1, -1, 1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, -1, -1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, 0, -1])
    light.set_energy(200)
    light.set_type("POINT")
    light.set_location([1, 0, 1])
    light.set_energy(200)

    # load the objects into the scene
    cad_id, model_id = cad_path.split("/")[-4:-2]
    shapeNet_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(cad_path)))
    )
    obj = bproc.loader.load_shapenet(
        shapeNet_path,
        used_synset_id=cad_id,
        used_source_id=model_id,
        move_object_origin=True,
    )
    if tless_like:
        # Check if the object has materials assigned
        obj.clear_materials()
        obj.new_material("tless_like")
        mat = obj.get_materials()[0]
        grey_col = np.random.uniform(0.2, 0.4)
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
        
    # obj.set_origin(mode="CENTER_OF_VOLUME")  # CENTER_OF_MASS
    import bpy

    # print(dir(obj))
    obj.select()
    # bpy.data.objects['model_normalized'].select = True
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    # # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    # bpy.ops.object.select_all(action='DESELECT')

    obj.set_cp("category_id", 1)
    # activate normal and distance rendering
    bproc.renderer.enable_distance_output(True)
    # set the amount of samples, which should be used for the color rendering
    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.set_output_format(enable_transparency=True)
    for name_pose in obj_poses:
        data_pose = obj_poses[name_pose]
        for idx_frame, pose in enumerate(data_pose):
            obj.set_local2world_mat(pose)
            data = bproc.renderer.render()
            rgb = Image.fromarray(np.uint8(data["colors"][0])).convert("RGBA")
            name = f"{name_pose}_{idx_frame:06d}"
            rgb.save(os.path.join(output_dir, f"{name}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("query_pose_path", nargs="?", help="Query pose")
    parser.add_argument("reference_pose_path", nargs="?", help="Reference pose")
    parser.add_argument("templates_poses_path", nargs="?", help="Template poses")
    parser.add_argument("save_path", nargs="?", help="Path to save images")
    parser.add_argument("gpu_id", nargs="?", help="GPUs id")
    parser.add_argument("texture", nargs="?", help="tless_like or not")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    args = parser.parse_args()
    tless_like = True if args.texture == "tless_like" else False
    os.makedirs(args.save_path, exist_ok=True)
    poses = {
        "query": np.load(args.query_pose_path),
        "reference": np.load(args.reference_pose_path),
        "templates": np.load(args.templates_poses_path),
    }
    intrinsic = np.array([[525, 0.0, 256], [0.0, 525, 256], [0.0, 0.0, 1.0]])

    img_size = [512, 512]
    os.makedirs(args.save_path, exist_ok=True)
    if args.disable_output == "true":
        # redirect output to log file
        logfile = args.save_path + "/render.log"
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
    # scale_meter do not change the binary mask but recenter_origin change it
    render_blender_proc(
        args.cad_path,
        args.save_path,
        poses,
        intrinsic=intrinsic,
        img_size=img_size,
        tless_like=tless_like,
    )
    if args.disable_output == "true":
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
