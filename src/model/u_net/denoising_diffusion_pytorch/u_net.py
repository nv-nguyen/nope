import inspect
from torch import nn
import torch
import math
import logging
import torch.nn.functional as F
from functools import partial
from diffusers import AutoencoderKL
import pytorch_lightning as pl
from src.model.u_net.denoising_diffusion_pytorch.model_utils import (
    exists,
    default,
    Attention,
    LinearAttention,
    Residual,
    PreNorm,
    Upsample,
    HardUpsample,
    Downsample,
    HardDownsample,
    ResnetBlock,
)
import logging


class UNet(pl.LightningModule):
    def __init__(
        self,
        u_net_dim,
        rot_representation_dim,
        encoder,
        pose_mlp_name,
        init_dim=None,
        out_dim=None,
        use_hard_up_down=True,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        **kwargs,
    ):
        super().__init__()
        logging.info("Initializing U-Net")

        # load pretrained backbone
        self.encoder = encoder
        self.channels = self.encoder.latent_dim
        self.name = self.encoder.name

        classes_dim = u_net_dim * 4
        init_dim = default(init_dim, u_net_dim)
        self.out_dim = default(out_dim, self.channels)
        dims = [init_dim, *map(lambda m: u_net_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_hard_up_down:
            downsample_klass = HardDownsample
            upsample_klass = HardUpsample
        else:
            downsample_klass = Downsample
            upsample_klass = Upsample

        # define pose encoder
        self.rot_representation_dim = rot_representation_dim
        if pose_mlp_name == "single_layer":
            self.pose_mlp = nn.Sequential(
                nn.Linear(rot_representation_dim, classes_dim),
            )
        elif pose_mlp_name == "two_layers":
            self.pose_mlp = nn.Sequential(
                nn.Linear(rot_representation_dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim),
            )
        elif pose_mlp_name == "posEncoding":
            from src.model.utils import SinusoidalPosEmb
            assert classes_dim % 6 == 0, "classes_dim must be divisible by 6 (rotation6d)"
            self.pose_mlp = SinusoidalPosEmb(dim=int(classes_dim // 6))
        self.init_conv = nn.Conv2d(self.channels, init_dim, 3, padding=1)
        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            time_emb_dim=classes_dim,
        )
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=classes_dim,
                        ),
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=classes_dim,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        downsample_klass(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block1 = block_klass(
            mid_dim,
            mid_dim,
            time_emb_dim=classes_dim,
        )

        self.mid_block2 = block_klass(
            mid_dim,
            mid_dim,
            time_emb_dim=classes_dim,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=classes_dim,
                        ),
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=classes_dim,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        upsample_klass(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        out_dim = default(out_dim, self.channels)
        self.final_res_block = block_klass(
            u_net_dim * 2,
            u_net_dim,
            time_emb_dim=classes_dim,
        )
        self.final_conv = nn.Sequential(
            block_klass(u_net_dim, u_net_dim),
            nn.Conv2d(u_net_dim, self.channels, 1),
        )
        logging.info("Intializing UNet done!")

    def forward(self, x, pose):
        x = self.init_conv(x)
        r = x.clone()

        # make class embeddings
        c = self.pose_mlp(pose)

        # define h for storing the intermediate outputs and use them for skip connections
        h = []
        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, c)
            h.append(x)
            x = block2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)
        # bottleneck
        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)
        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, c)
            x = attn(x)

            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, c)

        pred = self.final_conv(x)
        return pred


if __name__ == "__main__":
    from rich import print
    from src.model.encoder.AutoencoderKL import VAE_StableDiffusion

    vae = VAE_StableDiffusion(
        "/home/nguyen/Documents/pretrained/stable-diffusion-v1-5_vae.pth"
    ).cuda()
    model = UNet(
        u_net_dim=64,
        rot_representation_dim=6,
        encoder=vae,
    ).cuda()

    x1 = torch.randn(8, 4, 32, 32).cuda()
    classes = torch.rand((8, 6)).cuda()
    output = model(x1, classes)
    print(output.shape)
