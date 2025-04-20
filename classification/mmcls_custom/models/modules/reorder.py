import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule
from mmcls.models.builder import MODELS


class PatchEmbed(BaseModule):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_channels=3, embed_dims=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size - patch_size) // stride + 1, (img_size - patch_size) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dims) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Score(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Score, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(), # 或 nn.ReLU()
            # nn.Dropout(0.1), # 可选：防止过拟合
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class SoftSort(BaseModule):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores: torch.Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat
    
    
@MODELS.register_module()
class Reorder(BaseModule):
    def __init__(self, finetune_cfg):
        super().__init__()
        self.img_size=finetune_cfg.img_size
        self.patch_size=finetune_cfg.patch_size
        self.stride=finetune_cfg.stride
        self.in_channels=finetune_cfg.in_channels
        self.embed_dims=finetune_cfg.embed_dims
        self.num_patches=finetune_cfg.num_patches

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            stride=self.stride,
            in_channels=self.in_channels,
            embed_dims=self.embed_dims
        )

        self.score = Score(
            in_channels=self.embed_dims,
            hidden_channels=self.embed_dims * 2,
            out_channels=1
        )

        self.soft_sort = SoftSort(tau=1.0, hard=True, pow=1.0)

    def reorder_image(self, x, P):
        """
        将图像分成小块
        - x: (B, C, H, W) 的图像张量
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)   # B, C, H/ps, W/ps, ps, ps
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()    # B, H/ps, W/ps, C, ps, ps
        x = x.reshape(B, (H // self.patch_size) * (W // self.patch_size), C, self.patch_size, self.patch_size).contiguous()   # B, num_patches, C, ps, ps

        x_flat = x.reshape(B, self.num_patches, -1).contiguous()  # B, num_patches, C*ps*ps
        # print("x_flat.shape", x_flat.shape)
        # print("P.shape", P.shape)
        # P = P.squeeze(-1)
        x = torch.bmm(P, x_flat).reshape(B, int(np.sqrt(self.num_patches)), int(np.sqrt(self.num_patches)), C, self.patch_size, self.patch_size).contiguous()  # B, num_patches, C*ps*ps
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(B, C, H, W).contiguous()
        return x

    def forward(self, x):
        # get permutation matrix
        # print("x.shape", x.shape)
        embed_image = self.patch_embed(x)
        # print("embed_image.shape", embed_image.shape)
        s = self.score(embed_image)
        # print("s.shape", s.shape)
        P = self.soft_sort(s)
        # print("P.shape", P.shape)

        # sort the original image
        x = self.reorder_image(x, P).contiguous()
        # print("x_re.shape", x_re.shape)

        return x, P
