from torchvision import transforms, utils
from torch.utils import data
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import logging, os
import os.path as osp
from tqdm import tqdm
import time
import pandas as pd
import random
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
)

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
from src.utils.inout import load_json, save_json, casting_format_to_save_json
from src.poses.utils import (
    combine_R_and_T,
    get_obj_poses_from_template_level,
    crop_frame,
)
from src.utils.trimesh_utils import load_mesh
import cv2
import trimesh
from src.poses.utils import crop_frame

# set level logging
logging.basicConfig(level=logging.INFO)


class BaseBOP(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        **kwargs,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split

    def load_list_scene(self, split=None):
        if isinstance(split, str):
            if split is not None:
                split_folder = osp.join(self.root_dir, split)
            self.list_scenes = sorted(
                [
                    osp.join(split_folder, scene)
                    for scene in os.listdir(split_folder)
                    if os.path.isdir(osp.join(split_folder, scene))
                    and scene != "models"
                ]
            )
        elif isinstance(split, list):
            self.list_scenes = []
            for scene in split:
                if not isinstance(scene, str):
                    scene = f"{scene:06d}"
                if os.path.isdir(osp.join(self.root_dir, scene)):
                    self.list_scenes.append(osp.join(self.root_dir, scene))
            self.list_scenes = sorted(self.list_scenes)
        else:
            raise NotImplementedError
        logging.info(f"Found {len(self.list_scenes)} scenes")

    def load_scene(self, path, use_visible_mask=True):
        # Load rgb and mask images
        rgb_paths = sorted(Path(path).glob("rgb/*.png"))
        if use_visible_mask:
            mask_paths = sorted(Path(path).glob("mask_visib/*.png"))
        else:
            mask_paths = sorted(Path(path).glob("mask/*.png"))
        # load poses
        scene_gt = load_json(osp.join(path, "scene_gt.json"))
        scene_gt_info = load_json(osp.join(path, "scene_gt_info.json"))
        scene_camera = load_json(osp.join(path, "scene_camera.json"))
        return {
            "rgb_paths": rgb_paths,
            "mask_paths": mask_paths,
            "scene_gt": scene_gt,
            "scene_gt_info": scene_gt_info,
            "scene_camera": scene_camera,
        }

    def load_metaData(self, reset_metaData, mode="query", split="test", level=2):
        start_time = time.time()
        if mode == "query":
            metaData = {
                "scene_id": [],
                "frame_id": [],
                "obj_id": [],
                "idx_obj": [],
                "pose": [],
                "rgb_path": [],
                "mask_path": [],
                "mask_visib_path": [],
                "depth_path": [],
                "visib_fract": [],
                "bbox_obj": [],
                "bbox_visib": [],
                "intrinsic": [],
                "idx_template": [],
                "inplane": [],
            }
            logging.info(f"Loading metaData for split {split}")
            metaData_path = osp.join(self.root_dir, f"{split}_metaData.json")
            if reset_metaData:
                for scene_path in tqdm(self.list_scenes, desc="Loading metaData"):
                    scene_id = scene_path.split("/")[-1]
                    rgb_paths = sorted(Path(scene_path).glob("rgb/*.png"))
                    mask_paths = sorted(Path(scene_path).glob("mask/*.png"))
                    mask_paths = [str(x) for x in mask_paths]
                    mask_visib_paths = sorted(Path(scene_path).glob("mask_visib/*.png"))
                    mask_visib_paths = [str(x) for x in mask_visib_paths]
                    depth_paths = sorted(Path(scene_path).glob("depth/*.png"))
                    depth_paths = [str(x) for x in depth_paths]
                    video_metaData = {}

                    # load poses
                    for json_name in ["scene_gt", "scene_gt_info", "scene_camera"]:
                        json_path = osp.join(scene_path, json_name + ".json")
                        if osp.exists(json_path):
                            video_metaData[json_name] = load_json(json_path)
                        else:
                            video_metaData[json_name] = None
                    # load templates metaData
                    templates_metaData = load_json(
                        osp.join(scene_path, f"template_level{level}.json")
                    )
                    for idx_frame in range(len(rgb_paths)):
                        # get rgb path
                        rgb_path = rgb_paths[idx_frame]
                        # get id frame
                        id_frame = int(str(rgb_path).split("/")[-1].split(".")[0])
                        # templates data
                        templates_frame = templates_metaData[f"{id_frame}"]
                        # get frame gt
                        frame_gt = video_metaData["scene_gt"][f"{id_frame}"]
                        obj_ids = [int(x["obj_id"]) for x in frame_gt]
                        obj_poses = np.array(
                            [
                                combine_R_and_T(x["cam_R_m2c"], x["cam_t_m2c"])
                                for x in frame_gt
                            ]
                        )

                        for idx_obj in range(len(obj_ids)):
                            obj_id = obj_ids[idx_obj]
                            obj_pose = obj_poses[idx_obj]
                            mask_path = osp.join(
                                scene_path, "mask", f"{id_frame:06d}_{idx_obj:06d}.png"
                            )
                            mask_scene_path = osp.join(
                                scene_path, "mask", f"{id_frame:06d}.png"
                            )
                            mask_visib_path = osp.join(
                                scene_path,
                                "mask_visib",
                                f"{id_frame:06d}_{idx_obj:06d}.png",
                            )
                            depth_path = osp.join(
                                scene_path, "depth", f"{id_frame:06d}.png"
                            )
                            if mask_path in mask_paths:
                                metaData["mask_path"].append(mask_path)
                            elif mask_scene_path in mask_paths:
                                metaData["mask_path"].append(mask_scene_path)
                            else:
                                metaData["mask_path"].append(None)
                            if mask_visib_path in mask_visib_paths:
                                metaData["mask_visib_path"].append(mask_visib_path)
                            else:
                                metaData["mask_visib_path"].append(None)
                            if depth_path in depth_paths:
                                metaData["depth_path"].append(depth_path)
                            else:
                                metaData["depth_path"].append(None)
                            metaData["scene_id"].append(scene_id)
                            metaData["frame_id"].append(id_frame)
                            metaData["obj_id"].append(obj_id)
                            metaData["idx_obj"].append(idx_obj)
                            metaData["pose"].append(obj_pose)
                            metaData["rgb_path"].append(str(rgb_path))
                            metaData["intrinsic"].append(
                                video_metaData["scene_camera"][f"{id_frame}"]["cam_K"]
                            )
                            metaData["visib_fract"].append(
                                video_metaData["scene_gt_info"][f"{id_frame}"][idx_obj][
                                    "visib_fract"
                                ]
                                if "visib_fact"
                                in video_metaData["scene_gt_info"][f"{id_frame}"][
                                    idx_obj
                                ]
                                else 1.0
                            )
                            metaData["bbox_obj"].append(
                                video_metaData["scene_gt_info"][f"{id_frame}"][idx_obj][
                                    "bbox_obj"
                                ]
                                if "bbox_obj"
                                in video_metaData["scene_gt_info"][f"{id_frame}"][
                                    idx_obj
                                ]
                                else None
                            )
                            metaData["bbox_visib"].append(
                                video_metaData["scene_gt_info"][f"{id_frame}"][idx_obj][
                                    "bbox_visib"
                                ]
                                if "bbox_visib"
                                in video_metaData["scene_gt_info"][f"{id_frame}"][
                                    idx_obj
                                ]
                                else None
                            )
                            metaData["idx_template"].append(
                                templates_frame[idx_obj]["idx_template"]
                            )
                            metaData["inplane"].append(
                                templates_frame[idx_obj]["inplane"]
                            )
                            assert templates_frame[idx_obj]["obj_id"] == obj_id, print(
                                "Error mismatch object_id"
                            )
                # casting format of metaData
                metaData = casting_format_to_save_json(metaData)
                save_json(metaData_path, metaData)
            else:
                metaData = load_json(metaData_path)
        elif mode == "template":
            list_obj_ids, list_idx_template = [], []
            for obj_id in self.obj_ids:
                for idx_template in range(len(self.templates_poses)):
                    list_obj_ids.append(obj_id)
                    list_idx_template.append(idx_template)
            metaData = {
                "obj_id": list_obj_ids,
                "idx_template": list_idx_template,
            }

        self.metaData = pd.DataFrame.from_dict(metaData, orient="index")
        self.metaData = self.metaData.transpose()
        # shuffle data
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index(
            drop=True
        )
        finish_time = time.time()
        logging.info(
            f"Finish loading metaData of size {len(self.metaData)} in {finish_time - start_time:.2f} seconds"
        )
        return

    def open_PIL(self, path):
        return Image.open(path)

    def convert_rotation_representation(self, rot):
        if self.rot_representation == "rotation6d":
            return matrix_to_rotation_6d(rot)
        elif self.rot_representation == "euler_angles":
            return matrix_to_euler_angles(rot)
        elif self.rot_representation == "quaternion":
            return matrix_to_quaternion(rot)
        else:
            print("Not implemented!")

    def load_cad(self, cad_name="models"):
        cad_dir = f"{self.root_dir}/models/{cad_name}"
        cad_names = sorted(
            [
                x
                for x in os.listdir(cad_dir)
                if x.endswith(".ply") and not x.endswith("_old.ply")
            ]
        )
        models_info = load_json(osp.join(cad_dir, "models_info.json"))
        self.cads = {}
        for cad_name in cad_names:
            cad_id = int(cad_name.split(".")[0].replace("obj_", ""))
            cad_path = osp.join(cad_dir, cad_name)
            if os.path.exists(cad_path):
                mesh = load_mesh(cad_path)
            else:
                logging.warning("CAD model unavailable")
                mesh = None
            self.cads[cad_id] = {
                "mesh": mesh,
                "model_info": models_info[f"{cad_id}"]
                if f"{cad_id}" in models_info
                else models_info[cad_id],
            }
        logging.info(f"Loaded {len(cad_names)} models for dataset done!")
        return self.cads

    def get_template_path(self, template_dir, idx):
        obj_id = self.metaData.iloc[idx]["obj_id"]
        idx_template = self.metaData.iloc[idx]["idx_template"]
        path = osp.join(template_dir, f"obj_{obj_id:06d}/{idx_template:06d}.png")
        return path, idx_template

    def check_scene(self, scene_id, save_path):
        os.makedirs(save_path, exist_ok=True)
        # keep metada of scene_id
        metaData_scene = self.metaData[self.metaData["scene_id"] == scene_id]
        selected_frames = random.sample(list(metaData_scene["frame_id"].values), 5)
        colors = np.random.randint(0, 254, (len(self.cads), 3), dtype=np.uint8)

        for frame_id in selected_frames:
            frame_data = metaData_scene[metaData_scene["frame_id"] == frame_id]
            # read image
            rgb_path = frame_data["rgb_path"].values[0]
            cvImg = cv2.imread(str(rgb_path))
            # read openCV poses
            frame_poses = frame_data["pose"].values
            # get cad ids and use cad data project pts back on image
            cad_ids = frame_data["obj_id"].values
            obj_pcds = [
                trimesh.sample.sample_surface(self.cads[cad_id]["mesh"], 500)[0]
                for cad_id in cad_ids
            ]
            K = np.array(frame_data["intrinsic"].values[0]).reshape(3, 3)
            for idx_cad, cad_id in enumerate(cad_ids):
                pose = np.array(frame_poses[idx_cad]).reshape(4, 4)
                # project to image
                cvImg = render_pts_to_image(
                    cvImg=cvImg,
                    meshPts=obj_pcds[idx_cad],
                    K=K,
                    openCV_obj_pose=pose,
                    color=colors[idx_cad],
                )
            # save image
            vis_frame_path = f"{save_path}/{scene_id}_{frame_id}.png"
            logging.info("Saving visualization to {}".format(vis_frame_path))
            cv2.imwrite(vis_frame_path, cvImg)

    def crop_with_gt_pose(self, img, mask, pose, K, virtual_bbox_size):
        return crop_frame(
            img,
            mask,
            intrinsic=K,
            openCV_pose=pose,
            image_size=self.img_size,
            keep_inplane=False,
            virtual_bbox_size=virtual_bbox_size,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_dir = "/gpfsscratch/rech/xjd/uyb58rn/datasets/template-pose"
    dataset_names = ["hb", "hope", "icbin", "lm", "o-lm", "ruapc", "tudl"]
    # tless is special
    for dataset_name, split in zip(
        ["tless/test", "tless/train"], ["test_primesense", "train_primesense"]
    ):
        dataset = BaseBOP(os.path.join(root_dir, dataset_name), split)
        dataset.load_list_scene(split=split)
        dataset.load_metaData(reset_metaData=True)
        dataset.load_cad(cad_name="models_cad")
        for scene_path in dataset.list_scenes:
            scene_id = scene_path.split("/")[-1]
            dataset.check_scene(scene_id, f"./tmp/{dataset_name}")

    for dataset_name in tqdm(dataset_names):
        splits = [
            split
            for split in os.listdir(os.path.join(root_dir, dataset_name))
            if os.path.isdir(os.path.join(root_dir, dataset_name, split))
        ]
        splits = [
            split
            for split in splits
            if split.startswith("train")
            or split.startswith("val")
            or split.startswith("test")
        ]
        for split in splits:
            dataset = BaseBOP(os.path.join(root_dir, dataset_name), split)
            dataset.load_list_scene(split=split)
            dataset.load_metaData(reset_metaData=True)
            dataset.load_cad()
            for scene_path in dataset.list_scenes:
                scene_id = scene_path.split("/")[-1]
                dataset.check_scene(scene_id, f"./tmp/{dataset_name}")
