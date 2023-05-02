import importlib
import math
from torch import nn
import torch


def normalize_to_neg_one_to_one(img):
    tmp = img.clone()
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    img = t.clone()
    img = (img + 1) * 0.5
    return img.clamp(0, 1)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_dim=None):
        super().__init__()
        self.dim = dim
        self.max_dim = max_dim
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, None, :]  # Bxdim to BxdimxposEnc_size
        emb = emb.reshape(x.shape[0], -1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.max_dim is None:
            return emb
        else:
            return emb[:, :self.max_dim]

if __name__ == "__main__":
    from rich import print
    from src.model.encoder.AutoencoderKL import VAE_StableDiffusion

    mapping = SinusoidalPosEmb(60)
    classes = torch.rand((8, 6)).cuda()
    output = mapping(classes)
    print(output.shape)
