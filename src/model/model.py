from tqdm import tqdm
import torch
import math
import numpy as np
from os import path
import os
from torch import nn
from einops import rearrange, reduce
import torch.nn.functional as F
from collections import namedtuple
from functools import partial
import pytorch_lightning as pl
from torchvision import transforms as T, utils
from src.utils.visualization_utils import (
    put_image_to_grid,
)
from torchvision import transforms, utils
import logging
from PIL import Image
import os.path as osp
from src.model.utils import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one
from src.model.loss import GeodesicError
from src.model.normal_kl_loss import DiagonalGaussianDistribution
import imageio
import wandb
import multiprocessing
import time
from src.poses.vsd import vsd_obj
from src.utils.inout import save_json, casting_format_to_save_json


class PoseConditional(pl.LightningModule):
    def __init__(
        self,
        u_net,
        optim_config,
        testing_config,
        save_dir,
        **kwargs,
    ):
        super().__init__()
        self.u_net = u_net

        # define output
        self.save_dir = save_dir

        # define optimization scheme
        self.lr = optim_config.lr
        self.weight_decay = optim_config.weight_decay
        self.warm_up_steps = optim_config.warm_up_steps
        self.use_inv_deltaR = optim_config.use_inv_deltaR
        self.optim_name = "AdamW"

        # define testing config
        self.testing_config = testing_config

        if optim_config.loss_type == "l1":
            self.loss = F.l1_loss
        elif optim_config.loss_type == "l2":
            self.loss = F.mse_loss
        self.metric = GeodesicError()
        # define wandb logger
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "media"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
        self.log_dir = os.path.join(save_dir, "predictions")
        # define cad_dir for vsd evaluation
        self.tless_cad_dir = None

    def warm_up_lr(self):
        for optim in self.trainer.optimizers:
            for pg in optim.param_groups:
                pg["lr"] = self.global_step / float(self.warm_up_steps) * self.lr
            if self.global_step % 50 == 0:
                logging.info(f"Step={self.global_step}, lr warm up: lr={pg['lr']}")

    def configure_optimizers(self):
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                self.lr,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 30, 50, 100], gamma=0.5
        )
        return [optimizer], [lr_scheduler]

    def compute_loss(self, pred, gt):
        if self.loss is not None:
            loss = self.loss(pred, gt, reduction="none")
            loss = reduce(loss, "b ... -> b (...)", "mean")
            return loss.mean()
        else:
            pred = DiagonalGaussianDistribution(pred)
            loss = pred.kl(other=gt)
            return loss.mean()

    def forward(self, query, reference, relativeR):
        query_feat = self.u_net.encoder.encode_image(query)
        reference_feat = self.u_net.encoder.encode_image(reference, mode="mode")
        pred_query_feat = self.u_net(reference_feat, relativeR)
        loss = self.compute_loss(pred_query_feat, query_feat)
        return loss

    @torch.no_grad()
    def sample(self, reference, relativeR):
        reference_feat = self.u_net.encoder.encode_image(reference, mode="mode")
        pred_query_feat = self.u_net(reference_feat, relativeR)
        if hasattr(self.u_net.encoder, "decode_latent") and callable(
            self.u_net.encoder.decode_latent
        ):
            pred_rgb = self.u_net.encoder.decode_latent(pred_query_feat)
            pred_rgb = unnormalize_to_zero_to_one(pred_rgb)
        else:
            pred_rgb = None
        return pred_query_feat, pred_rgb

    def training_step_single_dataloader(self, batch, data_name):
        query = batch["query"]
        reference = batch["reference"]
        relativeR = batch["relativeR"]
        relativeR_inv = batch["relativeR_inv"]

        loss = self.forward(query=query, relativeR=relativeR, reference=reference)
        if self.use_inv_deltaR:
            loss_inv = self.forward(
                query=reference, reference=query, relativeR=relativeR_inv
            )
            loss = (loss + loss_inv) / 2
        self.log(f"loss/train_{data_name}", loss)

        # visualize reconstruction under GT pose
        if self.global_step % 1000 == 0:
            _, pred_rgb = self.sample(reference=reference, relativeR=relativeR)
            if pred_rgb is not None:
                save_image_path = path.join(
                    self.save_dir,
                    f"media/reconst_step{self.global_step}_rank{self.global_rank}.png",
                )
                vis_imgs = [
                    unnormalize_to_zero_to_one(reference),
                    unnormalize_to_zero_to_one(batch["query"]),
                    pred_rgb,
                ]
                vis_imgs, ncol = put_image_to_grid(vis_imgs)
                vis_imgs_resized = vis_imgs.clone()
                vis_imgs_resized = F.interpolate(
                    vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
                )
                utils.save_image(
                    vis_imgs_resized,
                    save_image_path,
                    nrow=ncol * 4,
                )
                self.logger.experiment.log(
                    {f"reconstruction/train_{data_name}": wandb.Image(save_image_path)},
                )
                print(save_image_path)
        return loss

    def training_step(self, batch, idx):
        loss_dict = {}
        loss_sum = 0
        for idx_dataloader, data_name in enumerate(batch.keys()):
            if self.trainer.global_step < self.warm_up_steps and idx_dataloader == 0:
                self.warm_up_lr()
            elif self.trainer.global_step == self.warm_up_steps and idx_dataloader == 0:
                logging.info(f"Finished warm up, setting lr to {self.lr}")
            loss = self.training_step_single_dataloader(batch[data_name], data_name)
            loss_dict[data_name] = loss
            loss_sum += loss
        loss_avg = loss_sum / len(batch.keys())
        self.log("loss/train_avg", loss_avg)
        return loss_avg

    @torch.no_grad()
    def log_score(self, dict_scores, split_name):
        for key, value in dict_scores.items():
            self.log(
                f"{key}/{split_name}",
                value,
                sync_dist=True,
            )

    def generate_templates(
        self, reference, all_relativeR, gt_templates, visualize=False
    ):
        b, c, h, w = reference.shape
        num_templates = all_relativeR.shape[1]
        # keep all predicted features of template for retrieval later
        if hasattr(self.u_net.encoder, "decode_latent") and callable(
            self.u_net.encoder.decode_latent
        ):
            pred_templates = torch.zeros(
                (b, num_templates, c, h, w), device=reference.device
            )
        else:
            pred_templates = None
        pred_feat_templates = torch.zeros(
            (b, num_templates, self.u_net.encoder.latent_dim, int(h / 8), int(w / 8)),
            device=reference.device,
        )
        frames = []
        for idx_template in tqdm(range(0, num_templates)):
            # get output of sample
            if visualize:
                vis_imgs = [
                    unnormalize_to_zero_to_one(reference),
                    unnormalize_to_zero_to_one(gt_templates[:, idx_template]),
                ]
            pred_feat_i, pred_rgb_i = self.sample(
                reference=reference, relativeR=all_relativeR[:, idx_template, :]
            )
            pred_feat_templates[:, idx_template] = pred_feat_i
            if pred_rgb_i is not None:
                pred_templates[:, idx_template] = pred_rgb_i
                if visualize:
                    vis_imgs.append(pred_rgb_i.to(torch.float16))
                    save_image_path = path.join(
                        self.save_dir,
                        f"media/template{idx_template}_rank{self.global_rank}.png",
                    )
                    vis_imgs, ncol = put_image_to_grid(vis_imgs)
                    vis_imgs_resized = vis_imgs.clone()
                    vis_imgs_resized = F.interpolate(
                        vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
                    )
                    utils.save_image(
                        vis_imgs_resized,
                        save_image_path,
                        nrow=ncol * 4,
                    )
                    frame = np.array(Image.open(save_image_path))
                    frames.append(frame)
        if visualize:
            # write video of denoising process with imageio ffmpeg
            vid_path = path.join(
                self.save_dir,
                f"media/video_step{self.global_step}_rank{self.global_rank}.mp4",
            )
            imageio.mimwrite(vid_path, frames, fps=5, macro_block_size=8)
        else:
            vid_path = None
        return pred_feat_templates, pred_templates, vid_path

    def retrieval(self, query, template_feat):
        num_templates = template_feat.shape[1]
        if self.testing_config.similarity_metric == "l2":
            query_feat = self.u_net.encoder.encode_image(query, mode="mode")
            query_feat = query_feat.unsqueeze(1).repeat(1, num_templates, 1, 1, 1)

            distance = (query_feat - template_feat) ** 2
            distance = torch.norm(distance, dim=2)
            similarity = -distance.sum(axis=3).sum(axis=2)  # B x N

            # get top 5 nearest templates
            _, nearest_idx = similarity.topk(k=5, dim=1)  # B x 1
            return similarity, nearest_idx

    def eval_geodesic(self, batch, data_name, visualize=True, save_prediction=False):
        if not (
            hasattr(self.u_net.encoder, "decode_latent")
            and callable(self.u_net.encoder.decode_latent)
        ):
            visualize = False
            logging.info(f"Setting visualize=False!")
        print("eval_geodesic", visualize)
        # visualize same loss as training
        query = batch["query"]
        batch_size = query.shape[0]
        reference = batch["reference"]
        relativeR = batch["gt_relativeR"]
        loss = self.forward(query=query, relativeR=relativeR, reference=reference)
        self.log(f"loss/val_{data_name}", loss)

        if visualize:
            # visualize reconstruction under GT pose
            save_image_path = path.join(
                self.save_dir,
                f"media/reconst_step{self.global_step}_rank{self.global_rank}.png",
            )
            _, pred_rgb = self.sample(reference=reference, relativeR=relativeR)
            if pred_rgb is not None:
                vis_imgs = [
                    unnormalize_to_zero_to_one(reference),
                    unnormalize_to_zero_to_one(batch["query"]),
                    pred_rgb,
                ]
                vis_imgs, ncol = put_image_to_grid(vis_imgs)
                vis_imgs_resized = vis_imgs.clone()
                vis_imgs_resized = F.interpolate(
                    vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
                )
                utils.save_image(
                    vis_imgs_resized,
                    save_image_path,
                    nrow=ncol * 4,
                )
                self.logger.experiment.log(
                    {f"reconstruction/val_{data_name}": wandb.Image(save_image_path)},
                )
        # retrieval templates
        gt_templates = batch["gt_templates"]
        all_relativeR = batch["all_relativeR"]
        pred_feat, pred_rgb, vid_path = self.generate_templates(
            reference=reference,
            all_relativeR=all_relativeR,
            gt_templates=gt_templates,
            visualize=visualize,
        )
        if visualize and pred_rgb is not None:
            self.logger.experiment.log(
                {f"templates/val_{data_name}": wandb.Video(vid_path)},
            )
        similarity, nearest_idx = self.retrieval(query=query, template_feat=pred_feat)

        if visualize:
            # visualize prediction
            save_image_path = path.join(
                self.save_dir,
                f"media/retrieved_step{self.global_step}_rank{self.global_rank}.png",
            )
            retrieved_template = gt_templates[
                torch.arange(0, batch_size, device=query.device), nearest_idx[:, 0]
            ]
            vis_imgs = [
                unnormalize_to_zero_to_one(reference),
                unnormalize_to_zero_to_one(batch["query"]),
                unnormalize_to_zero_to_one(retrieved_template),
            ]
            vis_imgs, ncol = put_image_to_grid(vis_imgs)
            vis_imgs_resized = vis_imgs.clone()
            vis_imgs_resized = F.interpolate(
                vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
            )
            utils.save_image(
                vis_imgs_resized,
                save_image_path,
                nrow=ncol * 4,
            )
            self.logger.experiment.log(
                {f"retrieval/val_{data_name}": wandb.Image(save_image_path)},
            )
        template_poses = batch["template_poses"][0]
        error, acc = self.metric(
            predR=template_poses[nearest_idx],
            gtR=batch["query_pose"],
            symmetry=batch["symmetry"].reshape(-1),
        )
        self.log_score(acc, split_name=f"val_{data_name}")

        # save predictions
        if save_prediction:
            save_path = os.path.join(
                self.save_dir,
                "predictions",
                f"pred_step{self.global_step}_rank{self.global_rank}",
            )
            vis_imgs = vis_imgs.cpu().numpy()
            query_pose = batch["query_pose"].cpu().numpy()
            similarity = similarity.cpu().numpy()
            np.savez(
                save_path,
                vis_imgs=vis_imgs,
                query_pose=query_pose,
                similarity=similarity,
            )
            print(save_path)

    def load_mesh(self, cad_dir):
        import trimesh
        import pyrender

        logging.info("Loading cad to avoid haning in validation lightning (BUG to fix)")
        self.tless_cad = {}
        for obj_id in range(1, 31):
            cad_path = osp.join(cad_dir, f"obj_{obj_id:06d}.ply")
            mesh = trimesh.load_mesh(cad_path)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            self.tless_cad[obj_id] = mesh
        logging.info("Loading mesh done!")

    def eval_vsd(self, batch, data_name, save_path):
        # visualize same loss as training
        query = batch["query"]
        batch_size = query.shape[0]
        reference = batch["reference"]
        relativeR = batch["gt_relativeR"]
        loss = self.forward(query=query, relativeR=relativeR, reference=reference)
        self.log(f"loss/val_{data_name}", loss)

        # visualize reconstruction under GT pose
        save_image_path = path.join(
            self.save_dir,
            f"media/reconst_val_step{self.global_step}_rank{self.global_rank}.png",
        )
        _, pred_rgb = self.sample(reference=reference, relativeR=relativeR)
        vis_imgs = [
            unnormalize_to_zero_to_one(reference),
            unnormalize_to_zero_to_one(batch["query"]),
            pred_rgb,
        ]
        vis_imgs, ncol = put_image_to_grid(vis_imgs)
        vis_imgs_resized = vis_imgs.clone()
        vis_imgs_resized = F.interpolate(
            vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
        )
        utils.save_image(
            vis_imgs_resized,
            save_image_path,
            nrow=ncol * 4,
        )
        self.logger.experiment.log(
            {f"reconstruction/val_{data_name}": wandb.Image(save_image_path)},
        )
        # retrieval templates
        visualize = True if "gt_templates" in batch else False
        gt_templates = None if not visualize else batch["gt_templates"]
        all_relativeR = batch["all_relativeR"]
        pred_feat, pred_rgb, vid_path = self.generate_templates(
            reference=reference,
            all_relativeR=all_relativeR,
            gt_templates=gt_templates,
            visualize=visualize,
        )
        similarity, nearest_idx = self.retrieval(query=query, template_feat=pred_feat)

        # visualize prediction
        if visualize:
            save_image_path = path.join(
                self.save_dir,
                f"media/retrieved_val_step{self.global_step}_rank{self.global_rank}.png",
            )
            retrieved_template = gt_templates[
                torch.arange(0, batch_size, device=query.device), nearest_idx[:, 0]
            ]
            vis_imgs = [
                unnormalize_to_zero_to_one(reference),
                unnormalize_to_zero_to_one(batch["query"]),
                unnormalize_to_zero_to_one(retrieved_template),
            ]
            vis_imgs, ncol = put_image_to_grid(vis_imgs)
            vis_imgs_resized = vis_imgs.clone()
            vis_imgs_resized = F.interpolate(
                vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
            )
            utils.save_image(
                vis_imgs_resized,
                save_image_path,
                nrow=ncol * 4,
            )
            self.logger.experiment.log(
                {f"retrieval/val_{data_name}": wandb.Image(save_image_path)},
            )
            self.logger.experiment.log(
                {f"templates/val_{data_name}": wandb.Video(vid_path)},
            )
        # getting VSD scores
        evaluate_vsd = True
        if evaluate_vsd:
            template_poses = batch["template_poses"]
            retrieved_R = template_poses[
                torch.arange(0, batch_size, device=query.device)[:, None].repeat(
                    1, nearest_idx.shape[1]
                ),
                nearest_idx,
            ]
            vsd_eval_inputs = {}
            gt_translation = batch["query_translation"]
            query_pose = batch["query_pose"]

            vsd_eval_inputs["intrinsic"] = batch["intrinsic"].cpu().numpy()
            vsd_eval_inputs["depth_path"] = batch["depth_path"]
            vsd_eval_inputs["obj_id"] = batch["obj_id"].cpu().numpy()

            pred_poses = torch.cat(
                (
                    retrieved_R,
                    gt_translation[:, None, :, :].repeat(1, retrieved_R.shape[1], 1, 1),
                ),
                dim=3,
            )  # B x N x 3 x 4
            # adding [0, 0, 0, 1] to make it 4x4
            pred_poses = torch.cat(
                (
                    pred_poses,
                    torch.zeros(
                        (batch_size, retrieved_R.shape[1], 1, 4), device=query.device
                    ),
                ),
                dim=2,
            )
            pred_poses[:, :, 3, 3] = 1.0
            vsd_eval_inputs["pred_poses"] = pred_poses.cpu().numpy()

            gt_poses = torch.cat((query_pose, gt_translation), dim=2)
            gt_poses = torch.cat(
                (gt_poses, torch.zeros((batch_size, 1, 4), device=query.device)), dim=1
            )
            gt_poses[:, 3, 3] = 1.0
            vsd_eval_inputs["gt_poses"] = gt_poses.cpu().numpy()
            vsd_eval_inputs["mesh"] = [
                self.tless_cad[int(id)] for id in batch["obj_id"].cpu().numpy()
            ]
            pool = multiprocessing.Pool(processes=self.trainer.num_workers)
            vsd_obj_from_index = partial(vsd_obj, list_frame_data=vsd_eval_inputs)
            start_time = time.time()
            vsd_error = list(
                tqdm(
                    pool.imap_unordered(
                        vsd_obj_from_index, range(len(vsd_eval_inputs["gt_poses"]))
                    ),
                    total=len(vsd_eval_inputs["gt_poses"]),
                )
            )
            vsd_error = np.stack(vsd_error, axis=0)  # Bxk where k is top k retrieved
            finish_time = time.time()
            print(
                f"Total time to render at rank {self.global_rank}",
                finish_time - start_time,
            )
            final_scores = {}
            for k in [1, 3, 5]:
                best_vsd = np.min(vsd_error[:, :k], 1)
                final_scores[f"top {k}, vsd_median"] = np.median(best_vsd)
                for threshold in [0.3]:
                    vsd_acc = (best_vsd <= threshold) * 100.0
                    # same for median
                    final_scores[f"top {k}, vsd_scores {threshold}"] = np.mean(vsd_acc)
            self.log_score(final_scores, split_name=f"val_{data_name}")
            np.save(save_path, vsd_error[:, 0])
            print(vsd_error[:, 0], save_path)
            return final_scores

    def validation_step(self, batch, idx):
        for idx_dataloader, data_name in enumerate(batch.keys()):
            if data_name in ["shapeNet"]:
                self.eval_geodesic(batch[data_name], data_name)
            elif data_name in ["tless"]:
                self.eval_vsd(batch[data_name], data_name)

    def test_step(self, batch, idx_batch):
        for idx_dataloader, dataloader_name in enumerate(batch.keys()):
            data_name, category = dataloader_name.split("_")
            if data_name in ["tless"]:
                save_path = os.path.join(
                        self.log_dir, f"vsd_{category}_batch{idx_batch}_rank_{self.global_rank}.npy"
                    )
                self.eval_vsd(batch[dataloader_name], category, save_path=save_path)
            else:
                
                self.eval_geodesic(
                    batch[dataloader_name],
                    category,
                    visualize=True,
                    save_prediction=True,
                )

    def test_epoch_end_vsd(self, test_step_outputs):
        # test_step_outputs is a list of dictionaries
        final_scores = {}
        for key in test_step_outputs[0].keys():
            final_scores[key] = np.mean(([x[key] for x in test_step_outputs]))
        self.logger.log_table(
            "final",
            columns=[k for k in final_scores.keys()],
            data=[[v for v in final_scores.values()]],
        )
