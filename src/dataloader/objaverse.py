import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
import random
import json
import os
import math
import os.path as osp
from src.poses.utils import (
    get_obj_poses_from_template_level,
    load_index_level0_in_level2,
    crop_frame,
)
from tqdm import tqdm
import logging

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
)
import glob
from src.utils.inout import write_txt, open_txt, save_json, load_json
from src.utils.shapeNet_utils import train_cats, test_cats, get_shapeNet_mapping
from pytorch_lightning import seed_everything
from src.utils.trimesh_utils import load_mesh, get_bbox_from_mesh, get_obj_diameter

seed_everything(2023)


class Objaverse(Dataset):
    def __init__(
        self,
        root_dir,
        img_size=256,
        rot_representation="rotation6d",
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.load_metaData()
        self.load_all_cad_paths()
        # define intrinsic
        self.img_size = img_size
        self.rot_representation = rot_representation
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Resize(self.img_size),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )

    def __len__(self):
        return len(self.query_paths)

    def load_metaData(self, max_obj=1000000):
        img_path_files = osp.join(self.root_dir, "img_path.txt")
        if not osp.exists(img_path_files):
            logging.info("Creating metaData ...")
            query_paths = []
            count_obj, count_not_valid = 0, 0
            for obj_id in tqdm(range(max_obj)):
                img_paths = glob.glob(
                    osp.join(self.root_dir, "images", f"obj_{obj_id:06d}", "*.png")
                )
                if len(img_paths) == 10:
                    query_paths += img_paths
                    count_obj += 1
                else:
                    count_not_valid + 1
            logging.info(
                f"Number of valid objects: {count_obj}, non valid objects {count_not_valid}"
            )
            random.shuffle(query_paths)
            write_txt(img_path_files, query_paths)
        self.query_paths = open_txt(img_path_files)

    def crop(self, img, pose):
        """
        This cropping can be removed if the CAD models are normalized to same scale during the rendering
        """
        intrinsic = np.array([[525, 0, 256], [0, 525, 256], [0, 0, 1]])
        # cropping with virtual bounding box
        img_cropped = crop_frame(
            img,
            mask=None,
            intrinsic=intrinsic,
            openCV_pose=pose,
            image_size=self.img_size,
            virtual_bbox_size=1,
        )
        return img_cropped

    def open_image(self, path):
        img = Image.open(path)
        mask = img.getchannel("A")
        black_bkg = Image.new("RGB", img.size, (0, 0, 0))
        black_bkg.paste(img, mask=mask)
        return black_bkg

    def convert_rotation_representation(self, rot):
        if self.rot_representation == "rotation6d":
            return matrix_to_rotation_6d(rot)
        elif self.rot_representation == "euler_angles":
            return matrix_to_euler_angles(rot)
        elif self.rot_representation == "quaternion":
            return matrix_to_quaternion(rot)
        else:
            print("Not implemented!")

    def sample_reference(self, query_path):
        obj_dir = osp.dirname(query_path)
        avail_img = glob.glob(osp.join(obj_dir, "*.png"))
        avail_img.remove(query_path)
        return random.choice(avail_img)

    def compute_relative_pose(self, query_pose, ref_pose):
        relative = query_pose[:3, :3] @ np.linalg.inv(ref_pose)[:3, :3]
        relative = torch.tensor(relative, dtype=torch.float32)
        relative_inv = ref_pose[:3, :3] @ np.linalg.inv(query_pose)[:3, :3]
        relative_inv = torch.tensor(relative_inv, dtype=torch.float32)
        return self.convert_rotation_representation(
            relative
        ), self.convert_rotation_representation(relative_inv)

    def get_pose(self, img_path):
        obj_name = osp.basename(osp.dirname(img_path))
        pose_path = osp.join(self.root_dir, "object_poses", f"{obj_name}.npy")
        idx = os.path.basename(img_path).split(".")[0]
        return np.load(pose_path)[int(idx)]

    def process(self, query_path, reference_path):
        query = self.open_image(query_path)
        reference = self.open_image(reference_path)
        query_pose = self.get_pose(query_path)
        ref_pose = self.get_pose(reference_path)

        # do some cropping here
        query = self.crop(query, query_pose)
        reference = self.crop(reference, ref_pose)
        relative_pose, relative_pose_inv = self.compute_relative_pose(
            query_pose, ref_pose
        )
        return (
            self.img_transform(query),
            self.img_transform(reference),
            relative_pose,
            relative_pose_inv,
        )

    def __getitem__(self, index):
        query_path = self.query_paths[index]
        reference_path = self.sample_reference(query_path)

        query, reference, relative, relative_inv = self.process(
            query_path, reference_path
        )
        return {
            "query": query.permute(2, 0, 1),
            "reference": reference.permute(2, 0, 1),
            "relativeR": relative,
            "relativeR_inv": relative_inv,
        }

    def load_all_cad_paths(self):
        self.uids = open_txt(osp.join(self.root_dir, "filtered_uids.txt"))
        self.cad_paths = load_json(osp.join(self.root_dir, "object_paths.json"))

    def normalize_mesh(self, mesh, scale_scene=0.8):
        size = mesh.extents
        scale = scale_scene / np.max(size)
        # scale mesh
        mesh.apply_scale(scale)
        return mesh