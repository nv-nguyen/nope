import os, random
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
from src.dataloader.baseBOP import BaseBOP
import logging
import cv2
import os.path as osp
from tqdm import tqdm
import torchvision
from einops import rearrange
from src.poses.utils import (
    crop_frame,
    NearestTemplateFinder,
    get_obj_poses_from_template_level,
    load_index_level0_in_level2,
    inverse_transform,
)
import copy
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
)
from src.utils.inout import get_root_project, load_json

# set level logging
logging.basicConfig(level=logging.INFO)


class BOPDataset(BaseBOP):
    def __init__(
        self,
        root_dir,
        template_dir,
        split,
        obj_ids,
        img_size,
        virtual_bbox_size=None,
        reset_metaData=False,
        pose_distribution="upper",
        rot_representation="rotation6d",
        isTesting=False,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.template_dir = template_dir
        self.split = split
        self.rot_representation = rot_representation
        self.pose_distribution = pose_distribution

        self.img_size = img_size
        self.mask_size = 25 if img_size == 64 else int(img_size // 8)
        self.virtual_bbox_size = virtual_bbox_size

        if isinstance(obj_ids, str):
            obj_ids = [int(obj_id) for obj_id in obj_ids.split(",")]
            logging.info(f"ATTENTION: Loading {len(obj_ids)} objects!")
        self.load_list_scene(split=split)
        self.load_template_poses()
        self.load_cad(cad_name="models" if "tless" not in root_dir else "models_cad")
        self.load_metaData(
            reset_metaData=reset_metaData,
            mode="query",
        )
        self.obj_ids = (
            obj_ids
            if obj_ids is not None
            else np.unique(self.metaData["obj_id"]).tolist()
        )
        self.metaData.reset_index(inplace=True)
        if (
            self.split.startswith("train") or self.split.startswith("val")
        ) and not isTesting:
            # keep only 90% of the data for training for each object
            self.isTesting = False
            self.metaData = self.subsample(self.metaData, 90)
        elif self.split.startswith("test") or isTesting:
            self.isTesting = True
            self.metaData = self.subsample(self.metaData, 100)
        else:
            raise NotImplementedError
        self.update_metaData()
        if "tless" in self.template_dir:
            init_size = len(self.metaData)
            root_project = get_root_project()
            # for tless setting, we subsample the dataset by taking only images from metaData
            with open(
                f"{root_project}/src/dataloader/tless_bop19_test_list.json"
            ) as json_file:
                bop19_test_list = json.load(json_file)
            bop19_test_list = pd.DataFrame.from_dict(bop19_test_list, orient="index")
            bop19_test_list = bop19_test_list.transpose()
            selected_frames = np.zeros(len(self.metaData), dtype=bool)
            for i in tqdm(range(len(self.metaData)), desc="Subsampling MetaData..."):
                idx_data = [
                    int(self.metaData.iloc[i]["scene_id"]),
                    self.metaData.iloc[i]["frame_id"],
                ]
                selected_frames[i] = (bop19_test_list == idx_data).all(1).any()
            self.metaData = self.metaData[selected_frames]
            self.metaData.reset_index(inplace=True)
            logging.info(
                f"Subsampling from size {init_size} to size {len(self.metaData)} by taking only images of BOP"
            )

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Resize(self.img_size),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        logging.info(
            f"Length of dataloader: {self.__len__()} containing objects {np.unique(self.metaData['obj_id'])}"
        )
        # define nearest pose finder
        self.nearest_pose_finder = NearestTemplateFinder(
            level_templates=2,
            pose_distribution="upper",
            return_inplane=True,
        )
        self.neighbor_poses = get_obj_poses_from_template_level(
            level=2, pose_distribution="all"
        )

    def load_template_poses(self):
        self.templates_poses = np.load(osp.join(self.template_dir, "obj_poses.npy"))
        # self.testing_indexes = load_index_level0_in_level2(self.pose_distribution)
        (
            self.testing_indexes,
            self.testing_templates_poses,
        ) = get_obj_poses_from_template_level(
            level=2, pose_distribution=self.pose_distribution, return_index=True
        )

    def subsample(self, df, percentage):
        # subsample the data for training and validation
        avail_obj_id = np.unique(df["obj_id"])
        selected_obj_id = [id for id in self.obj_ids]
        logging.info(f"Available {avail_obj_id}, selected {selected_obj_id} ")
        selected_index = []
        index_dataframe = np.arange(0, len(df))
        for obj_id in selected_obj_id:
            if self.isTesting:  # take all even occluded
                selected_index_obj = index_dataframe[df["obj_id"] == obj_id]
            else:
                selected_index_obj = index_dataframe[
                    np.logical_and(df["obj_id"] == obj_id, df["visib_fract"] >= 0.95)
                ]
            if percentage > 50:
                selected_index_obj = selected_index_obj[
                    : int(percentage / 100 * len(selected_index_obj))
                ]  # keep first
            else:
                selected_index_obj = selected_index_obj[
                    int((1 - percentage / 100) * len(selected_index_obj)) :
                ]  # keep last
            selected_index.extend(selected_index_obj.tolist())
        df = df.iloc[selected_index]
        logging.info(
            f"Subsampled from {len(index_dataframe)} to {len(df)} ({percentage}%) images"
        )
        return df

    def update_metaData(self):
        if self.pose_distribution == "upper":
            selected_idx = []
            for idx in range(len(self.metaData)):
                obj_pose = np.array(self.metaData.iloc[idx]["pose"]).reshape(4, 4)
                cam_pose = inverse_transform(obj_pose)
                if cam_pose[2, 3] >= 0.0:
                    selected_idx.append(idx)
            # keep only selected objects
            init_len = len(self.metaData)
            self.metaData = self.metaData.iloc[selected_idx].reset_index(drop=True)
            logging.info(
                f"Update metaData from length {init_len} to length {len(self.metaData)}!"
            )

    def __len__(self):
        return len(self.metaData)

    def crop(self, img, pose, intrinsic, diameter):
        """
        This cropping can be removed if the CAD models are normalized to same scale during the rendering
        """
        # cropping with virtual bounding box
        virtual_bbox_size = (
            diameter * 1.2 if self.virtual_bbox_size is None else self.virtual_bbox_size
        )
        # convert everything to meter to make it consistent with crop_frame function
        tmp_pose = np.copy(pose)
        tmp_pose[:3, 3] /= 1000.0
        virtual_bbox_size /= 1000.0

        img_cropped = crop_frame(
            np.array(img),
            mask=None,
            # keep_inplane=False,
            intrinsic=intrinsic,
            openCV_pose=tmp_pose,
            image_size=self.img_size,
            virtual_bbox_size=virtual_bbox_size,
        )
        return img_cropped

    def load_image(self, df, idx):
        rgb_path = df.iloc[idx]["rgb_path"]
        rgb = Image.open(rgb_path).convert("RGB")
        mask_path = df.iloc[idx]["mask_path"]
        mask = Image.open(mask_path)
        if len(np.array(mask).shape) == 3:
            if np.array(mask).shape[2] == 3:
                mask = np.array(mask)[:, :, 0]
                mask = Image.fromarray(np.uint8(mask))

        # remove background
        black_img = Image.new("RGB", rgb.size, (0, 0, 0))
        black_img.paste(rgb, mask)
        rgb = black_img

        intrinsic = np.array(df.iloc[idx]["intrinsic"]).reshape(3, 3)
        pose = np.array(df.iloc[idx]["pose"]).reshape(4, 4)
        obj_id = df.iloc[idx]["obj_id"]
        diameter = self.cads[obj_id]["model_info"]["diameter"]
        rgb_cropped = self.crop(rgb, pose, intrinsic, diameter)
        return rgb_cropped, pose

    def decompose_pose(self, pose):
        idx_neighbour, inplane = self.nearest_pose_finder.search_nearest_template(
            pose.reshape(-1, 4, 4)
        )
        updated_pose = self.neighbor_poses[idx_neighbour[0]]
        updated_pose[:3, 3] = pose[:3, 3]
        return inplane[0], updated_pose

    def convert_rotation_representation(self, rot):
        if self.rot_representation == "rotation6d":
            return matrix_to_rotation_6d(rot)
        elif self.rot_representation == "euler_angles":
            return matrix_to_euler_angles(rot)
        elif self.rot_representation == "quaternion":
            return matrix_to_quaternion(rot)
        else:
            print("Not implemented!")

    def compute_relative_pose(self, query_pose, ref_pose):
        relative = query_pose[:3, :3] @ np.linalg.inv(ref_pose)[:3, :3]
        relative = torch.tensor(relative, dtype=torch.float32)
        relative_inv = ref_pose[:3, :3] @ np.linalg.inv(query_pose)[:3, :3]
        relative_inv = torch.tensor(relative_inv, dtype=torch.float32)
        return self.convert_rotation_representation(
            relative
        ), self.convert_rotation_representation(relative_inv)

    def __getitem__(self, idx):
        query, query_pose = self.load_image(self.metaData, idx)
        # sample a reference image
        new_frame = copy.copy(self.metaData)
        obj_id = self.metaData.iloc[idx]["obj_id"]
        scene_id = self.metaData.iloc[idx]["scene_id"]
        same_obj_and_same_scene_not_occluded = np.logical_and(
            new_frame.obj_id == obj_id, new_frame.scene_id == scene_id
        )
        same_obj_and_same_scene_not_occluded = np.logical_and(
            same_obj_and_same_scene_not_occluded, new_frame.visib_fract >= 0.95
        )
        if np.sum(same_obj_and_same_scene_not_occluded) == 0:
            same_obj_and_same_scene_not_occluded = np.logical_and(
                new_frame.obj_id == obj_id, new_frame.visib_fract >= 0.95
            )
            logging.info("Same scene_id does not exist!")
        new_frame = new_frame[same_obj_and_same_scene_not_occluded].reset_index(
            drop=True
        )
        idx_relative = np.random.randint(0, len(new_frame))
        reference, reference_pose = self.load_image(new_frame, idx_relative)

        # remove inplane rotation (however, it can be used during testing )
        _, query_pose_wo_inp = self.decompose_pose(query_pose)
        _, reference_pose_wo_inp = self.decompose_pose(reference_pose)

        relativeR, relativeR_inv = self.compute_relative_pose(
            query_pose_wo_inp, reference_pose_wo_inp
        )
        if not self.isTesting:
            return {
                "query": self.img_transform(query).permute(2, 0, 1),
                "reference": self.img_transform(reference).permute(2, 0, 1),
                "relativeR": relativeR,
                "relativeR_inv": relativeR_inv,
            }
        else:
            raise NotImplementedError
