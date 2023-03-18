import torch
import torch.nn as nn
from Swin_Transformer_Block import SwinTransformerBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.swin_transformer_opt import Swin_Transformer_Type

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, heads, window_size, mlp_ratio, drop, drop_path=0, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build_blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=heads, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 drop=drop,
                                 drop_path=drop_path)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim, input_resolution)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chan=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)

        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]]

        self.img_size = img_size
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[0]

        self.proj = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layerNorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchMerging(nn.Module):
  def __init__(self, dim, input_resolution):
    super().__init__()
    self.input_resolution = input_resolution
    self.dim = dim
    self.reduction = nn.Linear(4 * dim, 2* dim, bias=False)
    self.norm = nn.LayerNorm(4 * dim)

  def forward(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

    x = self.norm(x)
    x = self.reduction(x)

    return x


class Swin_Transformer(nn.Module):
    def __init__(self, training_type, downsample_type, patch_size, img_size=224, num_classes=1000, mlp_ratio=4.,
                 drop_rate=0.):
        super().__init__()

        # dataloader = DataLoader(dataset, )
        SType = Swin_Transformer_Type(training_type, img_size, downsample_type)
        dim, heads, depth, stochastic_depth = SType.setting_downsampling()
        window_size = SType.image_size_option()

        self.num_classes = num_classes  # int
        self.dim = dim  # int
        self.img_size = img_size  # int
        self.s_depth = stochastic_depth  # int
        self.num_layers = len(depth)  # list

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chan=3, embed_dim=dim
        )
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_features = int(dim * 2 ** (self.num_layers - 1))
        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(0.)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depth))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depth[i_layer],
                               heads=heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               drop=drop_rate,
                               drop_path=self.s_depth,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)

            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  # dropout

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

