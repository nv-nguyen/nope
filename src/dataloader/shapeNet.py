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
import zipfile
from io import BytesIO

seed_everything(2023)


class ShapeNet(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        pose_distribution="upper",
        rot_representation="quaternion",
        fast_evaluation=False,
        img_size=256,
        level=2,
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.rot_representation = rot_representation
        self.pose_distribution = pose_distribution
        self.fast_evaluation = fast_evaluation
        self.level = level
        self.load_testing_template_poses()

        self.load_symmetry_mapping()
        self.load_metaData()
        logging.info(f"Length of dataset: query={len(self.query_paths)}")

        # define intrinsic
        self.img_size = img_size
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Resize(self.img_size),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )

    def __len__(self):
        return len(self.query_paths)

    def get_index_in_pose_distribution(self):
        self.indexes, _ = get_obj_poses_from_template_level(
            level=self.level,
            pose_distribution=self.pose_distribution,
            return_index=True,
        )

    def list_template_paths(self, obj_path):
        paths = [
            os.path.join(obj_path, f"templates_{idx:06d}.png") for idx in self.indexes
        ]
        return paths

    def get_img_from_paths(self, paths, load_all=False):
        # get index of pose distribution
        self.get_index_in_pose_distribution()
        all_images = []
        for path in paths:
            all_images += glob.glob(
                osp.join(self.root_dir, "images", path, "query_*.png")
            )
            if (
                self.split == "training" and load_all
            ):  # if training, dont care about that is query, reference or template
                all_images += glob.glob(
                    osp.join(self.root_dir, "images", path, "reference_*.png")
                )
                all_images += self.list_template_paths(
                    osp.join(self.root_dir, "images", path)
                )
        random.shuffle(all_images)
        return all_images

    def load_metaData(self):
        """
        There are three different splits:
        1. Training sets: ~1000 cads per category (with 13 categories in total)
        2. Unseen instances sets: 50 cads per category (with 13 categories in total)
        3. Unseen categories sets: 1000 cads per category (with 10 categories in total)
        """
        self.is_testing_split = False if self.split == "training" else True
        id2cat_mapping, cat2id_mapping = get_shapeNet_mapping()
        selected_cats = (
            train_cats
            if self.split in ["training", "unseen_training"]
            else [self.split]
        )
        # get all obj ids
        selected_obj_ids = {cat: [] for cat in selected_cats}
        all_cad_names = open_txt(osp.join(self.root_dir, "cad_names.txt"))
        for obj_id, cad_name in enumerate(all_cad_names):
            cat_obj = id2cat_mapping[cad_name.split("_")[0]]
            if cat_obj not in selected_cats:
                continue
            selected_obj_ids[cat_obj].append(obj_id)

        # shuffle obj ids for each category for a given seed, then sample based on split
        all_obj_ids = []
        for cat in selected_cats:
            random.shuffle(selected_obj_ids[cat])
            if self.split == "training":
                all_obj_ids.extend(selected_obj_ids[cat][50:])
            elif self.split == "unseen_training":
                all_obj_ids.extend(selected_obj_ids[cat][:50])
            else:
                all_obj_ids.extend(selected_obj_ids[cat][:100])

        self.query_paths = []
        self.query_to_references = {}
        load_all = True if self.split == "training" else False
        for obj_id in tqdm(all_obj_ids, desc="Loading data"):
            obj_path = f"{self.root_dir}/images/obj_{obj_id:06d}"
            if not os.path.exists(obj_path):
                logging.warning(f"Path {obj_path} does not exist")
                continue
            self.query_paths.append(obj_path)
            self.query_to_references[f"obj_{obj_id:06d}"] = self.get_img_from_paths(
                [obj_path], load_all=load_all
            )
        self.query_paths = self.get_img_from_paths(self.query_paths)

    def load_symmetry_mapping(self):
        id2cat_mapping, cat2id_mapping = get_shapeNet_mapping()

        self.all_cad_names = open_txt(osp.join(self.root_dir, "cad_names.txt"))
        cat_ids = [
            id2cat_mapping[cad_name.split("_")[0]] for cad_name in self.all_cad_names
        ]
        self.obj_name2symmetry = {}
        for idx, cat_id in enumerate(cat_ids):
            self.obj_name2symmetry[f"obj_{idx:06d}"] = 2 if cat_id in ["bottle"] else 0

    def crop(self, img, pose):
        # TODO: remove this function by normalize all CAD to same scale and put norm(translation)=1
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
        try:
            img = Image.open(path)
        except:
            # handle corrupted images
            logging.info(f"Image {path} is corrupted, try to open with zipfile")
            obj_id = int(path.split("/")[-2][4:])
            obj_id_segment = obj_id // 300
            zip_path = os.path.join(
                self.root_dir,
                f"zip/{obj_id_segment*300:06d}_to_{(obj_id_segment+1)*300:06d}.zip",
            )
            archive = zipfile.ZipFile(zip_path, "r")
            imgfile = archive.open(
                os.path.join(path.split("/")[-2], path.split("/")[-1])
            )
            # Read the contents of the file into a memory buffer
            data = BytesIO(imgfile.read())
            # Open the image data with PIL
            img = Image.open(data)
            img.save(path)
            logging.info(f"{path} is fixed!")
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
        if not self.is_testing_split:
            obj_name = osp.basename(osp.dirname(query_path))
            return random.choice(self.query_to_references[obj_name])
        else:
            obj_dir = osp.dirname(query_path)
            ref_paths = glob.glob(osp.join(obj_dir, "reference*.png"))
            return random.choice(ref_paths)

    def get_pose(self, path):
        obj_name = osp.basename(osp.dirname(path))
        filename = osp.basename(path)
        type = filename.split("_")[0]
        if type == "templates":
            type = "template"  # dirty fix
        idx = int(filename.split("_")[1].split(".")[0])
        pose = np.load(
            osp.join(self.root_dir, f"object_{type}_poses", obj_name + ".npy")
        )[idx]
        return pose

    def compute_relative_pose(self, query_pose, ref_pose):
        relative = query_pose[:3, :3] @ np.linalg.inv(ref_pose)[:3, :3]
        relative = torch.tensor(relative, dtype=torch.float32)
        relative_inv = ref_pose[:3, :3] @ np.linalg.inv(query_pose)[:3, :3]
        relative_inv = torch.tensor(relative_inv, dtype=torch.float32)
        return self.convert_rotation_representation(
            relative
        ), self.convert_rotation_representation(relative_inv)

    def load_testing_template_poses(self):
        # load poses
        level = 0 if self.fast_evaluation else 2
        (
            self.testing_indexes,
            self.testing_templates_poses,
        ) = get_obj_poses_from_template_level(
            level=level, pose_distribution=self.pose_distribution, return_index=True
        )
        # load indexes templates
        if self.fast_evaluation and self.level == 2:
            self.testing_indexes = load_index_level0_in_level2(self.pose_distribution)

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
        if not self.is_testing_split:
            # convert to tensor
            return (
                self.img_transform(query),
                self.img_transform(reference),
                relative_pose,
                relative_pose_inv,
            )
        else:
            # read images
            obj_path = osp.dirname(query_path)
            all_templates = [
                self.open_image(f"{obj_path}/templates_{idx:06d}.png")
                for idx in self.testing_indexes
            ]
            all_templates = [
                self.img_transform(
                    self.crop(
                        all_templates[idx],
                        self.get_pose(f"{obj_path}/templates_{idx:06d}.png"),
                    )
                )
                for idx in range(len(all_templates))
            ]
            # read relative pose
            all_relativeR = [
                self.compute_relative_pose(self.testing_templates_poses[idx], ref_pose)[
                    0
                ]
                for idx in range(len(all_templates))
            ]

            query_pose = torch.from_numpy(self.get_pose(query_path))
            template_poses = torch.from_numpy(self.testing_templates_poses)
            return (
                self.img_transform(query),
                self.img_transform(reference),
                relative_pose,
                torch.stack(all_templates),
                torch.stack(all_relativeR),
                query_pose,
                template_poses,
            )

    def get_symmetry(self, query_path):
        obj_name = osp.basename(osp.dirname(query_path))
        return torch.Tensor([self.obj_name2symmetry[obj_name]])

    def __getitem__(self, index):
        query_path = self.query_paths[index]
        reference_path = self.sample_reference(query_path)
        if not self.is_testing_split:
            query, reference, relative, relative_inv = self.process(
                query_path, reference_path
            )
            return {
                "query": query.permute(2, 0, 1),
                "reference": reference.permute(2, 0, 1),
                "relativeR": relative,
                "relativeR_inv": relative_inv,
            }
        else:
            (
                query,
                reference,
                gt_relativeR,
                gt_templates,
                all_relativeR,
                query_pose,
                template_poses,
            ) = self.process(query_path, reference_path)
            return {
                "query": query.permute(2, 0, 1),
                "reference": reference.permute(2, 0, 1),
                "gt_relativeR": gt_relativeR,
                "all_relativeR": all_relativeR,
                "gt_templates": gt_templates.permute(0, 3, 1, 2),
                "symmetry": self.get_symmetry(query_path),
                "query_pose": query_pose[:3, :3],
                "template_poses": template_poses[:, :3, :3],
            }