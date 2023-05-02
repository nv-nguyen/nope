import logging
import os
import numpy as np
from tqdm import tqdm
import random
from src.poses.utils import get_obj_poses_from_template_level, get_root_project
from src.utils.trimesh_utils import get_obj_diameter
from src.utils.inout import write_txt, open_txt, save_json, load_json
import os.path as osp
from functools import partial
import multiprocessing
import time
from src.poses.utils import look_at, spherical_to_cartesian, inverse_transform
from src.utils.shapeNet_utils import train_cats, test_cats, get_shapeNet_mapping
import argparse
import glob
import pytorch_lightning as pl
import objaverse

pl.seed_everything(2023)
random.seed(2023)
np.random.seed(2023)


def filter_obj(ids, annotations):
    filtered_ids = []
    for id in tqdm(ids):
        having_category = len(annotations[id]["categories"]) == 1
        if having_category:
            filtered_ids.append(id)
    return filtered_ids


def generate_query_and_reference_poses(idx, save_pose_paths, num_poses=10, radius=1.0):
    """
    Generating camera query poses and reference poses
    """
    azimuths = np.random.uniform(0, 2 * np.pi, num_poses)
    elevetions = np.random.uniform(0, np.pi / 2, num_poses)

    # convert to cartesian coordinates
    location = spherical_to_cartesian(azimuths, elevetions, radius)
    center_points = np.zeros_like(location)
    query_poses = np.zeros((num_poses, 4, 4))

    for idx_pose in range(num_poses):
        tmp = look_at(location[idx_pose], center_points[idx_pose])
        query_poses[idx_pose] = np.copy(inverse_transform(tmp))
        norm = np.linalg.norm(query_poses[idx_pose, :3, 3])

        if np.abs(norm - radius) > 0.1:
            logging.warning(f"Warning: location {norm} is bigger than radius {radius}")
    np.save(save_pose_paths[idx], query_poses)


def call_blender(
    idx,
    list_cad_paths,
    list_save_paths,
    list_poses_paths,
    disable_output,
    custom_blender_path,
):
    cad_path = list_cad_paths[idx]
    save_path = list_save_paths[idx]
    obj_poses_path = list_poses_paths[idx]
    gpu_ids = idx % 4
    os.makedirs(save_path, exist_ok=True)
    command = f"{custom_blender_path}/blender -b --python src/poses/blender_objaverse.py -- --cad_path {cad_path} --pose_path {obj_poses_path} --output_dir {save_path} --gpu_id {gpu_ids}"
    if disable_output:
        command += " --disable_output"
    # disable output when running os.system
    if disable_output:
        command += " > /dev/null 2>&1"
    # print(command)
    os.system(command)
    # count number of images
    num_imgs = len(glob.glob(osp.join(save_path, "*.png")))
    return num_imgs == 10, command


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("ShapeNet dataset generation scripts")
    parser.add_argument(
        "--cad_dir",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    parser.add_argument(
        "--custom_blender_path",
        type=str,
        help="Custom blender path",
    )
    parser.add_argument(
        "--disable_output", action="store_true", help="Disable output of blender"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=500)
    args = parser.parse_args()

    object_poses_dir = osp.join(args.save_dir, f"object_poses")
    object_img_dir = osp.join(args.save_dir, f"images")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(object_poses_dir, exist_ok=True)
    os.makedirs(object_img_dir, exist_ok=True)
    root_repo = get_root_project()

    logging.info("Loading object paths")
    if False:
        uids = objaverse.load_uids()
        object_paths = objaverse._load_object_paths()
        annotations = objaverse.load_annotations(uids)
        filtered_uids = filter_obj(uids, annotations)
        write_txt("./tmp/filtered_uids.txt", filtered_uids)
        save_json("./tmp/object_paths.json", object_paths)
    else:
        filtered_uids = open_txt(osp.join(args.save_dir, "filtered_uids.txt"))
        object_paths = load_json(osp.join(args.save_dir, "object_paths.json"))
    logging.info(
        f"Loading object paths {len(filtered_uids)},  {len(object_paths)} done!"
    )

    logging.info("Start generating poses")
    all_cad_paths, all_pose_paths, all_save_paths = [], [], []
    for idx in tqdm(range(args.start_index, args.end_index)):
        obj_bop_name = f"obj_{idx:06d}"  # convert to bop format
        object_poses_path = osp.join(object_poses_dir, obj_bop_name + ".npy")

        all_pose_paths.append(object_poses_path)
        all_save_paths.append(osp.join(object_img_dir, obj_bop_name))
        all_cad_paths.append(osp.join(args.cad_dir, object_paths[filtered_uids[idx]]))

    generate_query_and_reference_poses_with_index = partial(
        generate_query_and_reference_poses,
        save_pose_paths=all_pose_paths,
    )
    # generate poses
    pool = multiprocessing.Pool(processes=args.num_workers)
    start_time = time.time()
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                generate_query_and_reference_poses_with_index,
                range(len(all_pose_paths)),
            ),
            total=len(all_pose_paths),
        )
    )
    finish_time = time.time()
    logging.info(f"Total time to generate query pose {finish_time - start_time}")

    # rendering images
    call_blender_with_index = partial(
        call_blender,
        list_cad_paths=all_cad_paths,
        list_save_paths=all_save_paths,
        list_poses_paths=all_pose_paths,
        disable_output=args.disable_output,
        custom_blender_path=args.custom_blender_path,
    )
    # generate images
    start_time = time.time()
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                call_blender_with_index,
                range(len(all_save_paths)),
            ),
            total=len(all_save_paths),
        )
    )
    finish_time = time.time()
    logging.info(
        f"Number of suceessful renderings: {sum([1 for x in mapped_values if x[0]])}"
    )
    logging.info(f"Total time to generate images {finish_time - start_time}")
    logging.info("Step 2: Generating poses and images done!")
