from torch.utils.data import Dataset
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from functools import partial
import trimesh
import pyrender
from src.utils.inout import load_json
from src.poses.vsd_utils import (
    depth_im_to_dist_im_fast,
    estimate_visib_mask_gt,
    estimate_visib_mask_est,
)
import logging
import cv2
import os

import os
from PIL import Image

os.environ["DISPLAY"] = ":1"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def pyrenderer(
    obj_poses,
    BOP_cad_trimesh,
    intrinsic,
    img_size,
):
    # camera pose is fixed as np.eye(4)
    cam_pose = np.eye(4)
    # convert openCV camera
    cam_pose[1, 1] = -1
    cam_pose[2, 2] = -1
    # create scene config
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
    )
    # create camera and render engine
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000
    )
    scene.add(camera, pose=cam_pose)
    render_engine = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    cad_node = scene.add(BOP_cad_trimesh, pose=np.eye(4), name="cad")
    if len(obj_poses.shape) == 2:
        obj_poses = obj_poses[None, ...]
    depths = []
    for idx_frame in range(obj_poses.shape[0]):
        scene.set_pose(cad_node, obj_poses[idx_frame])
        depth = render_engine.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        depths.append(depth)
    return depths


def vsd_obj(
    idx_frame,
    list_frame_data,
    delta_vsd=15,
    tau_vsd=20,
    cost_type="step",
    use_gt_translation=True,
):
    # read frame data
    frame_data = list_frame_data[idx_frame]
    mesh = frame_data["mesh"]
    # mesh = trimesh.load_mesh(frame_data["cad_path"])
    # mesh = pyrender.Mesh.from_trimesh(mesh)
    # print("Loading mesh done!")
    cam_K = np.array(frame_data["intrinsic"]).reshape(3, 3)

    depth_test = cv2.imread(frame_data["depth_path"], -1) / 10.0
    image_size = depth_test.shape
    pred_poses = np.array(frame_data["pred_poses"]).reshape(-1, 4, 4)
    gt_poses = np.array(frame_data["query_pose"]).reshape(4, 4)
    gt_poses = np.tile(gt_poses[None, ...], (pred_poses.shape[0], 1, 1))
    renderer = partial(
        pyrenderer,
        BOP_cad_trimesh=mesh,
        intrinsic=cam_K,
        img_size=image_size,
    )
    # gt poses
    if use_gt_translation:
        pred_poses[:, :3, 3] = gt_poses[:, :3, 3]
    else:
        raise NotImplementedError
    gt_depths = renderer(gt_poses)
    pred_depths = renderer(pred_poses)
    vsd_error = np.zeros(len(gt_depths))
    for idx_poses in range(len(gt_depths)):
        depth_gt = gt_depths[idx_poses]
        depth_est = pred_depths[idx_poses]
        # Convert depth images to distance images.
        dist_test = depth_im_to_dist_im_fast(depth_test, cam_K)
        dist_gt = depth_im_to_dist_im_fast(depth_gt, cam_K)
        dist_est = depth_im_to_dist_im_fast(depth_est, cam_K)

        # Visibility mask of the model in the ground-truth pose.
        visib_gt = estimate_visib_mask_gt(
            dist_test, dist_gt, delta_vsd, visib_mode="bop19"
        )

        # Visibility mask of the model in the estimated pose.
        visib_est = estimate_visib_mask_est(
            dist_test, dist_est, visib_gt, delta_vsd, visib_mode="bop19"
        )

        # Intersection and union of the visibility masks.
        visib_inter = np.logical_and(visib_gt, visib_est)
        visib_union = np.logical_or(visib_gt, visib_est)

        visib_union_count = visib_union.sum()
        visib_comp_count = visib_union_count - visib_inter.sum()

        # Pixel-wise distances.
        dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])

        # Calculate VSD for each provided value of the misalignment tolerance.
        if visib_union_count == 0:
            vsd_error[idx_poses] = 1.0
        else:
            # Pixel-wise matching cost.
            if cost_type == "step":
                costs = dists >= tau_vsd
            elif cost_type == "tlinear":  # Truncated linear function.
                costs = dists / tau_vsd
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError("Unknown pixel matching cost.")
            vsd_error[idx_poses] = (np.sum(costs) + visib_comp_count) / float(
                visib_union_count
            )
    return vsd_error


if __name__ == "__main__":
    results = load_json("./tmp/test_vsd/pred_seen.json")
    for idx in range(len(results)):
        frame_pred = results[idx]
        depth_path = frame_pred["depth_path"]
        intrinsic = np.array(frame_pred["intrinsic"]).reshape(3, 3)
        depth_test = cv2.imread(depth_path, -1) / 10.0
        pred_poses = np.array(frame_pred["prediction"]).reshape(
            -1, 4, 4
        )  # "pred_poses"
        gt_poses = np.array(frame_pred["query_pose"]).reshape(4, 4)
        gt_poses = np.tile(gt_poses[None, ...], (pred_poses.shape[0], 1, 1))
        mesh = trimesh.load_mesh(frame_pred["cad_path"])
        mesh = pyrender.Mesh.from_trimesh(mesh)
        vsd_score = vsd_obj(
            mesh,
            pred_poses,
            gt_poses,
            depth_test,
            cam_K=intrinsic,
            image_size=depth_test.shape[:2],
        )
        print(vsd_score)
