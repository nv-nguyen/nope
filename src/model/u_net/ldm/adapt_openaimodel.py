from abc import abstractmethod

import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.model.u_net.ldm.openaimodel import UNetModel
import pytorch_lightning as pl


class UNetModelPose(UNetModel):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        injecting_condition_twice,
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
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
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
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,  # custom transformer support
            transformer_depth=transformer_depth,  # custom transformer support
            context_dim=context_dim,  # custom transformer support
            n_embed=n_embed,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=legacy,
        )
        self.time_embed_dim = model_channels * 4
        if pose_mlp_name == "single_layer":
            self.pose_mlp = nn.Sequential(
                nn.Linear(rot_representation_dim, context_dim),
            )
        elif pose_mlp_name == "two_layers":
            self.pose_mlp = nn.Sequential(
                nn.Linear(rot_representation_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim),
            )
        elif pose_mlp_name == "posEncoding":
            from src.model.utils import SinusoidalPosEmb
            if context_dim % 6 == 0:
                logging.warning(f"context_dim={context_dim} must be divisible by 6 (rotation6d)")
            self.pose_mlp = SinusoidalPosEmb(dim=int(context_dim // 6)+1, max_dim=context_dim)

        self.injecting_condition_twice = injecting_condition_twice
        if self.injecting_condition_twice: # map pose as time embedding
            self.pose_mlp_timesteps = nn.Sequential(
                nn.Linear(rot_representation_dim, self.time_embed_dim),
            )
        # load pretrained backbone
        self.encoder = encoder
        self.channels = self.encoder.latent_dim
        self.name = self.encoder.name

    def forward(self, x, pose):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # skipping timesteps
        if self.injecting_condition_twice:
            emb = self.pose_mlp_timesteps(pose)
        else:
            emb = torch.zeros(x.shape[0], self.time_embed_dim, device=x.device)
        context = self.pose_mlp(pose).unsqueeze(1)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


if __name__ == "__main__":
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    from src.utils.weight import load_checkpoint
    import logging

    logging.basicConfig(level=logging.INFO)

    cfg = OmegaConf.load("configs/model/cin_ldm_vq_f8_debug.yaml")
    model_path = "/home/nguyen/Documents/pretrained/ldm/model.ckpt"
    # model_path = "/gpfsscratch/rech/xjd/uyb58rn/pretrained/openai/256x256_diffusion.pt"
    u_net = instantiate(cfg.u_net).cuda()
    load_checkpoint(
        u_net, model_path, checkpoint_key="state_dict", prefix="model.diffusion_model."
    )
    x = torch.randn(8, 4, 32, 32).cuda()
    classes = torch.rand((8, 6)).cuda()
    output = u_net(x, classes)
    print(output.shape)
