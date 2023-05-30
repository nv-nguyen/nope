import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL


class VAE_StableDiffusion(pl.LightningModule):
    def __init__(
        self,
        pretrained_path,
        latent_dim=4,
        name="vae",
        using_KL=False,
        **kwargs,
    ):
        super().__init__()
        self.encoder = AutoencoderKL.from_config(f"{pretrained_path}/config.json")
        self.encoder.load_state_dict(
            torch.load(f"{pretrained_path}/diffusion_pytorch_model.bin")
        )
        self.latent_dim = latent_dim
        self.name = name
        self.using_KL = using_KL
        if self.using_KL:
            self.encode_mode = None
        else:
            self.encode_mode = "mode"

    @torch.no_grad()
    def encode_image(self, image, mode=None):
        mode = self.encode_mode if mode is None else mode
        with torch.no_grad():
            if mode == "mode":
                latent = self.encoder.encode(image).latent_dist.mode() * 0.18215
            elif mode is None:
                latent = self.encoder.encode(
                    image
                ).latent_dist  # DiagonalGaussianDistribution instance
                latent.mean *= 0.18215
            else:
                raise NotImplementedError
        return latent

    @torch.no_grad()
    def decode_latent(self, latent):
        latent = latent / 0.18215
        with torch.no_grad():
            return self.encoder.decode(latent).sample
