from typing import Optional
import torch
from torch import nn

from .transformer_layer import Transformer_ST_TDC_gra_sharp


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x, x)


# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
            self,
            image_size = None,
            patches: int = 4,
            dim: int = 96,
            ff_dim: int = 144,
            num_heads: int = 4,
            num_layers: int = 12,
            dropout_rate: float = 0.1,
            T: int = 160,
            theta: float = 0.7
    ):
        super().__init__()
        self.T = T
        self.dim = dim

        # Image and patch sizes
        # t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        # gt, gh, gw = t // ft, h // fh, w // fw  # number of patches
        # seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, (1, 5, 5), stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        # self.normLast = nn.LayerNorm(dim, eps=1e-6)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, (3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, (3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x, tau):

        b, c, t, fh, fw = x.shape  # B x C x T x H x W

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]

        x = self.patch_embedding(x)  # [B, 64, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]

        # score 用于可视化
        Trans_features, Score1 = self.transformer1(x, tau)  # [B, 4*4*40, 64]
        Trans_features2, Score2 = self.transformer2(Trans_features, tau)  # [B, 4*4*40, 64]
        Trans_features3, Score3 = self.transformer3(Trans_features2, tau)  # [B, 4*4*40, 64]

        # Trans_features3 = self.normLast(Trans_features3)

        # upsampling heads
        # features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 64, 40, 4, 4]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)  # [B, 64, 40, 4, 4]

        features_last = self.upsample(features_last)  # x [B, 64, 7*7, 80]
        features_last = self.upsample2(features_last)  # x [B, 32, 7*7, 160]

        features_last = torch.mean(features_last, 3)  # x [B, 32, 160, 4]
        features_last = torch.mean(features_last, 3)  # x [B, 32, 160]
        rPPG = self.ConvBlockLast(features_last)  # x [B, 1, 160]

        # pdb.set_trace()

        rPPG = rPPG.squeeze(1)  # B x T

        return rPPG, Score1, Score2, Score3
