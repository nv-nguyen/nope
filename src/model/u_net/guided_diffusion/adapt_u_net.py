from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.model.u_net.guided_diffusion.u_net import UNetModel


class UNetModelPose(UNetModel):
    def __init__(
        self,
        pose_mlp_name,
        rot_representation_dim,  # added
        encoder,  # added
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        **kwargs,
    ):
        UNetModel.__init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
        )
        time_embed_dim = model_channels * 4
        if pose_mlp_name == "single_layer":
            self.pose_mlp = nn.Sequential(
                nn.Linear(rot_representation_dim, time_embed_dim),
            )
        elif pose_mlp_name == "two_layers":
            self.pose_mlp = nn.Sequential(
                nn.Linear(rot_representation_dim, time_embed_dim),
                nn.GELU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        elif pose_mlp_name == "posEncoding":
            from src.model.utils import SinusoidalPosEmb

            if time_embed_dim % 6 != 0:
                logging.warning("u_net_dim must be divisible by 6 (rotation6d)")
            self.pose_mlp = SinusoidalPosEmb(dim=int(time_embed_dim // 6))

        # load pretrained backbone
        self.encoder = encoder
        self.channels = self.encoder.latent_dim
        self.name = self.encoder.name

    def forward(self, x, pose):
        hs = []
        emb = self.pose_mlp(pose)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, emb)
            hs.append(h)
        h = self.middle_block(h, emb, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, emb)
        h = h.type(x.dtype)
        return self.out(h)


if __name__ == "__main__":
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    import torch
    from src.utils.weight import load_checkpoint
    import logging

    logging.basicConfig(level=logging.INFO)

    cfg = OmegaConf.load("configs/model/guided_df_cf_256.yaml")
    model_path = "/home/nguyen/Documents/pretrained/openai/256x256_diffusion.pt"
    model_path = "/gpfsscratch/rech/xjd/uyb58rn/pretrained/openai/256x256_diffusion.pt"
    u_net = instantiate(cfg.u_net).cuda()
    load_checkpoint(u_net, model_path)
    # u_net.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    x = torch.randn(8, 4, 32, 32).cuda()
    classes = torch.rand((8, 6)).cuda()
    output = u_net(x, classes)
    print(output.shape)
