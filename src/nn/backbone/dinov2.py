import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from src.nn.backbone import vision_transformer as vits
from src.core import register
from src.utils import utils as dinov2_utils
# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa

dinov2_weights = 'models/dinov2_vitl14_pretrain.pth'

dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )

dinov2 = vits.__dict__['vit_large'](**dinov2_kwargs)
dinov2_utils.load_pretrained_weights(dinov2, dinov2_weights, "teacher")
dinov2.eval()

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
@register
class DINOv2EncoderViT(nn.Module):
    __inject__ = ['encoder',]
    def __init__(
        self,
        encoder: nn.Module,
        out_chans: int = 256,
        use_fc: bool = False
    ):
        super().__init__()
        #encoder = dinov2

        self.patch_size = encoder.patch_size
        self.encoder = encoder
        dinov2_utils.load_pretrained_weights(self.encoder, dinov2_weights, "teacher")
        self.encoder.eval()
        if not use_fc:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    encoder.embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )
        else:
            self.neck = nn.Conv2d(
                    encoder.embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                )
        for n, param in self.encoder.named_parameters():
            param.requires_grad = False

    def get_enc_embs(self, pixel_values: torch.FloatTensor):
        b, _, h, w = pixel_values.shape
        h, w = h // self.encoder.patch_size, w // self.encoder.patch_size
        image_embeddings = self.encoder.forward_features(pixel_values)["x_prenorm"][:, 1:]
        image_embeddings = image_embeddings.permute(0, 2, 1).contiguous().reshape(b, -1, h, w) # b, c, h, w

        return image_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.get_enc_embs(x)
        x = self.neck(x)
        # import pdb;pdb.set_trace()
        # # x = self.get_enc_embs(x)
        # x = self.neck(x)
        # pdb.set_trace()
        return x
