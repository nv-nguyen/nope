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

random.seed(2023)
np.random.seed(2023)


def select_cad_with_texture(idx, root, list_cats, cat2id_mapping, max_num_cad=10):
    cat = list_cats[idx]
    cat_dir = osp.join(root, cat2id_mapping[cat])
    available_cad_models = sorted(os.listdir(cat_dir))
    logging.info(f"Available CAD models: {len(available_cad_models)} for {cat}")

    list_cad_paths, list_cad_name, obj_diameter = [], [], {}
    for cad_model in tqdm(available_cad_models):
        cad_path = osp.join(cat_dir, cad_model, "models", "model_normalized.obj")
        texture_path = cad_path.replace("models/model_normalized.obj", "images")
        material_path = cad_path.replace(".obj", ".mtl")
        try:
            diameter = get_obj_diameter(cad_path)
        except:
            logging.warning(f"Error in getting diameter for {cad_path}")
            diameter = None
            continue
        having_texture = osp.exists(texture_path) or osp.exists(material_path)
        if having_texture and diameter is not None:
            list_cad_paths.append(cad_path)
            cad_name = osp.join(f"{cat2id_mapping[cat]}_{cad_model}")
            list_cad_name.append(cad_name)
            obj_diameter[cad_name] = diameter
            if len(list_cad_paths) >= max_num_cad:
                break
    return list_cad_paths, list_cad_name, obj_diameter


def generate_query_and_reference_poses(
    idx, save_query_paths, save_ref_paths, diameters, num_poses=5, radius=1.0
):
    """
    Generating camera query poses and reference poses
    """
    azimuths = np.random.uniform(0, 2 * np.pi, num_poses * 2)
    elevetions = np.random.uniform(0, np.pi / 2, num_poses * 2)

    # convert to cartesian coordinates
    location = spherical_to_cartesian(azimuths, elevetions, radius)
    center_points = np.zeros_like(location)

    query_poses = np.zeros((num_poses, 4, 4))
    ref_poses = np.zeros((num_poses, 4, 4))

    query_poses_norm1 = np.zeros((num_poses, 4, 4))
    ref_poses_norm1 = np.zeros((num_poses, 4, 4))

    for idx_pose in range(num_poses):
        tmp = look_at(location[2 * idx_pose], center_points[2 * idx_pose])
        query_poses_norm1[idx_pose] = np.copy(inverse_transform(tmp))
        tmp[:3, 3] *= 1.2 * diameters[idx]
        query_poses[idx_pose] = np.copy(inverse_transform(tmp))

        tmp = look_at(location[2 * idx_pose + 1], center_points[2 * idx_pose + 1])
        ref_poses_norm1[idx_pose] = np.copy(inverse_transform(tmp))
        tmp[:3, 3] *= 1.2 * diameters[idx]
        ref_poses[idx_pose] = np.copy(inverse_transform(tmp))

        norm = np.linalg.norm(query_poses_norm1[idx_pose, :3, 3])
        if np.abs(norm - radius) > 0.1:
            logging.warning(f"Warning: location {norm} is bigger than radius {radius}")

    np.save(save_query_paths[idx].replace(".npy", "_norm1.npy"), query_poses_norm1)
    np.save(save_ref_paths[idx].replace(".npy", "_norm1.npy"), ref_poses_norm1)

    np.save(save_query_paths[idx], query_poses)
    np.save(save_ref_paths[idx], ref_poses)


def call_blender(
    idx,
    list_cad_paths,
    list_save_paths,
    list_query_poses_paths,
    list_ref_poses_paths,
    list_templates_poses_path,
    disable_output,
    tless_like,
    gpu_id,
    custom_blender_path,
):
    cad_path = list_cad_paths[idx]
    save_path = list_save_paths[idx]
    query_pose_path = list_query_poses_paths[idx]
    ref_pose_path = list_ref_poses_paths[idx]
    templates_poses_path = list_templates_poses_path[idx]

    os.makedirs(save_path, exist_ok=True)
    command = f"blenderproc run src/poses/blenderproc.py {cad_path} {query_pose_path} {ref_pose_path} {templates_poses_path} {save_path} {gpu_id}"
    if tless_like:
        command += " tless_like"
    else:
        command += " no_tless_like"
    if disable_output:
        command += " true"
    if custom_blender_path is not None:
        command += f" --custom-blender-path {custom_blender_path}"
    # disable output when running os.system
    if disable_output:
        command += " > /dev/null 2>&1"
    print(command)
    # os.system(command)
    # count number of images
    num_imgs = len(glob.glob(osp.join(save_path, "*.png")))
    return num_imgs == 652, command


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
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="all, select_cad, generate_poses_and_images",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=20000)
    parser.add_argument(
        "--tless_like", action="store_true", help="Rendering texture less like T-LESS"
    )
    parser.add_argument(
        "--disable_output", action="store_true", help="Disable output of blender"
    )
    parser.add_argument("--level", type=int, default=2)
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=4)

    os.makedirs(args.save_dir, exist_ok=True)
    name_file_cad_paths = osp.join(args.save_dir, "cad_paths.txt")
    name_file_cad_names = osp.join(args.save_dir, "cad_names.txt")
    name_file_obj_diameter = osp.join(args.save_dir, "obj_diameter.json")

    all_cats = train_cats + test_cats
    id2cat_mapping, cat2id_mapping = get_shapeNet_mapping()
    root_repo = get_root_project()

    if args.step == "all" or args.step == "select_cad":
        select_cad_with_texture_with_index = partial(
            select_cad_with_texture,
            root=args.cad_dir,
            list_cats=all_cats,
            cat2id_mapping=cat2id_mapping,
        )
        # generate images
        start_time = time.time()
        mapped_values = list(
            tqdm(
                pool.imap_unordered(
                    select_cad_with_texture_with_index,
                    range(len(all_cats)),
                ),
                total=len(all_cats),
            )
        )
        finish_time = time.time()
        logging.info(f"Total time to select CAD models: {finish_time - start_time}")

        all_cad_paths, all_cad_names, dict_obj_diameter = [], [], None
        for data in tqdm(mapped_values):
            input_cads, save_paths, obj_diameter = data
            all_cad_paths.extend(input_cads)
            all_cad_names.extend(save_paths)
            if dict_obj_diameter is None:
                dict_obj_diameter = obj_diameter
            else:
                dict_obj_diameter.update(obj_diameter)
        logging.info(
            f"Input: {len(all_cad_paths)}, save {len(all_cad_names)}, dict_obj_diameter {len(dict_obj_diameter)}"
        )
        write_txt(name_file_cad_paths, all_cad_paths)
        write_txt(name_file_cad_names, all_cad_names)
        save_json(name_file_obj_diameter, dict_obj_diameter)
        logging.info("Step 1: Select CAD models with texture done!")

    if args.step == "all" or args.step == "generate_poses_and_images":
        object_template_poses = get_obj_poses_from_template_level(
            level=args.level, pose_distribution="all"
        )
        all_cad_paths = open_txt(name_file_cad_paths)
        all_cad_names = open_txt(name_file_cad_names)
        dict_obj_diameter = load_json(name_file_obj_diameter)

        object_query_poses_dir = osp.join(args.save_dir, f"object_query_poses")
        object_ref_poses_dir = osp.join(args.save_dir, f"object_reference_poses")
        object_template_poses_dir = osp.join(args.save_dir, f"object_template_poses")

        os.makedirs(object_query_poses_dir, exist_ok=True)
        os.makedirs(object_ref_poses_dir, exist_ok=True)
        os.makedirs(object_template_poses_dir, exist_ok=True)

        (
            all_query_pose_paths,
            all_ref_pose_paths,
            all_templates_poses_path,
            all_diameters,
            all_save_paths,
        ) = ([], [], [], [], [])

        logging.info(
            "Running rendering from index {} to {}".format(
                args.start_index, args.end_index
            )
        )
        args.end_index = min(args.end_index, len(all_cad_paths))
        # all_cad_paths = all_cad_paths[args.start_index : args.end_index]

        for idx in tqdm(range(args.start_index, args.end_index)):
            obj_bop_name = f"obj_{idx:06d}"  # convert to bop format
            object_query_pose_path = osp.join(
                object_query_poses_dir, f"{obj_bop_name}.npy"
            )
            object_ref_pose_path = osp.join(object_ref_poses_dir, f"{obj_bop_name}.npy")
            object_template_poses_path = osp.join(
                object_template_poses_dir, f"{obj_bop_name}.npy"
            )

            all_query_pose_paths.append(object_query_pose_path)
            all_ref_pose_paths.append(object_ref_pose_path)
            all_diameters.append(dict_obj_diameter[all_cad_names[idx]])
            all_save_paths.append(osp.join(args.save_dir, "images", obj_bop_name))

            # generate template poses
            norm = np.linalg.norm(object_template_poses[0, :3, 3])
            tmp = np.copy(object_template_poses)
            tmp[:, :3, 3] *= 1 / norm * 1.2 * all_diameters[-1]
            np.save(object_template_poses_path, tmp)
            all_templates_poses_path.append(object_template_poses_path)

        generate_query_and_reference_poses_with_index = partial(
            generate_query_and_reference_poses,
            save_query_paths=all_query_pose_paths,
            save_ref_paths=all_ref_pose_paths,
            diameters=all_diameters,
        )
        # generate images
        start_time = time.time()
        mapped_values = list(
            tqdm(
                pool.imap_unordered(
                    generate_query_and_reference_poses_with_index,
                    range(len(all_diameters)),
                ),
                total=len(all_diameters),
            )
        )
        finish_time = time.time()
        logging.info(f"Total time to generate query pose {finish_time - start_time}")

        call_blender_with_index = partial(
            call_blender,
            list_cad_paths=all_cad_paths[args.start_index : args.end_index],
            list_save_paths=all_save_paths,
            list_query_poses_paths=all_query_pose_paths,
            list_ref_poses_paths=all_ref_pose_paths,
            list_templates_poses_path=all_templates_poses_path,
            disable_output=args.disable_output,
            tless_like=args.tless_like,
            gpu_id=args.gpu_ids,
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

        # collect failed renderings to re-render
        list_fails = []
        for idx, value in enumerate(mapped_values):
            if not value:
                list_fails.append(idx)
        write_txt(
            osp.join(
                args.save_dir,
                f"failed_renderings_{args.start_index}_{args.end_index}.txt",
            ),
            list_fails,
        )
        logging.info(f"Total time to generate images {finish_time - start_time}")
        logging.info("Step 2: Generating poses and images done!")
