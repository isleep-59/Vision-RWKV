# my_modules/finetune_vrwkv.py
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, load_checkpoint
from mmcls.models.builder import BACKBONES, build_backbone


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size - patch_size) // stride + 1, (img_size - patch_size) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Score(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Score, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        nn.Linear(in_channels, out_channels, bias=False)


class SoftSort(nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores: torch.Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat
    

@BACKBONES.register_module()
class OrderFinetuneVRWKV(BaseModule):
    def __init__(self, finetune_cfg, backbone_cfg, backbone_ckpt=None, init_cfg=None):
        super().__init__(init_cfg)

        self.patch_embed = PatchEmbed(
            img_size=finetune_cfg.img_size,
            patch_size=finetune_cfg.patch_size,
            stride=finetune_cfg.stride,
            in_chans=finetune_cfg.in_channels,
            embed_dim=finetune_cfg.embed_dims
        )

        self.score = Score(
            in_channels=finetune_cfg.embed_dims,
            out_channels=1
        )

        self.soft_sort = SoftSort(tau=1.0, hard=False, pow=1.0)

        backbone_cfg['img_size'] = finetune_cfg.img_size
        backbone_cfg['patch_size'] = finetune_cfg.patch_size
        backbone_cfg['embed_dims'] = finetune_cfg.num_patches
        self.backbone = build_backbone(backbone_cfg)

        if backbone_ckpt is not None:
            load_checkpoint(self.backbone, backbone_ckpt, map_location='cpu')
            print(f"✅ Loaded backbone checkpoint from {backbone_ckpt}")

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def forward(self, x):
        x = self.score(x)
        x = self.backbone(x)
        return x
