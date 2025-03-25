import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
# from xlstmutils import PatchEmbed, VitPatchEmbed, VitPosEmbed2d, PatchMerging, PatchExpand, FinalPatchExpand_X4
try :
    from vision_lstm.vision_lstm2 import *
except:
    from .vision_lstm.vision_lstm2 import *
from timm.layers import drop_path, DropPath, to_2tuple, trunc_normal_, to_ntuple
import einops
import warnings

import numpy as np
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_, to_ntuple

from torchinfo import summary
from calflops import calculate_flops


# from kappamodules.functional.pos_embed import interpolate_sincos
def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


class PatchMergingv2(nn.Module):
    r""" Patch Merging Layer for 2D (generalization to 3D is analogous).

    This layer reshapes a flat patch sequence (B, L, C) into (B, H, W, C)
    where (H, W) = input_resolution, then partitions each spatial axis into blocks
    of 2 (i.e. merging 4 patches), concatenates along the channel dimension, applies
    normalization, and a linear reduction mapping the concatenated channels to (2 * C).

    Args:
        input_resolution (tuple[int]): Resolution of input grid, e.g. (H, W).
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # e.g. (H, W)
        self.dim = dim
        self.ndim = len(input_resolution)
        # For 2D, merging 2x2 patches means 4 patches merged.
        self.num_patches_merged = 2 ** self.ndim  # 4 for 2D, 8 for 3D
        # Reduction maps (merged_channels = num_patches_merged * dim) --> (2 * dim)
        self.reduction = nn.Linear(self.num_patches_merged * dim, 2 * dim, bias=False)
        self.norm = norm_layer(self.num_patches_merged * dim)

    def forward(self, x):
        B, L, C = x.shape
        expected_L = int(np.prod(self.input_resolution))
        assert L == expected_L, f"input feature has wrong size: got {L}, expected {expected_L}"
        # new spatial resolution after merging: each dimension halved
        new_res = [r // 2 for r in self.input_resolution]  # e.g. [28, 28] for 56x56
        # Reshape from (B, L, C) to (B, H, W, C)
        x = x.view(B, *self.input_resolution, C)
        # Rearrange into an intermediate shape: (B, r0, 2, r1, 2, C)
        # Here r0 = new_res[0], r1 = new_res[1]
        x = einops.rearrange(
            x,
            "b (r0 2) (r1 2) c -> b r0 2 r1 2 c",
            r0=new_res[0], r1=new_res[1]
        )
        # Merge the two constant axes with the channel axis: (B, r0, r1, 2*2*C)
        x = x.reshape(B, new_res[0], new_res[1], 4 * C)
        # Flatten spatial dimensions: (B, r0, r1, 4*C) -> (B, r0*r1, 4*C)
        x = x.reshape(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.ndim = len(input_resolution)
        self.num_patches_merged = 2 ** self.ndim  # e.g. 4 for 2D, 8 for 3D
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

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


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(self.ndim)(patch_size)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(self.ndim)(stride)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0, \
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            # use primitive type as np.prod gives np.int which is not compatible with all serialization/logging
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            # initialize as nn.Linear
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)), \
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x

# from kappamodules.vit import VitPosEmbed2d
class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        return x + embed
    


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


from termcolor import colored
def _print(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{bcolors.ENDC}"
  
    if "bold" in p.lower():
        pre += bcolors.BOLD
    elif "underline" in p.lower():
        pre += bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += bcolors.HEADER
      
    if "warning" in p.lower():
        pre += bcolors.WARNING
    elif "error" in p.lower():
        pre += bcolors.FAIL
    elif "ok" in p.lower():
        pre += bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += bcolors.OKBLUE
        else: 
            pre += bcolors.OKCYAN

    print(f"{pre}{string}{bcolors.ENDC}")



import yaml
def load_config(config_filepath):
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)
    

    
import numpy as np
from matplotlib import pyplot as plt
def show_sbs(im1, im2, figsize=[8,4], im1_title="Image", im2_title="Mask", show=True):
    if im1.shape[0]<4:
        im1 = np.array(im1)
        im1 = np.transpose(im1, [1, 2, 0])
        
    if im2.shape[0]<4: 
        im2 = np.array(im2)
        im2 = np.transpose(im2, [1, 2, 0])
        
    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(im1)
    axs[0].set_title(im1_title)
    axs[1].imshow(im2, cmap='gray')
    axs[1].set_title(im2_title)
    if show: plt.show()


class MyViLBlockEnc(nn.Module):
    """ A basic Vision xLSTM layer for one stage in encoder.
    """
    def __init__(self,
                 embed_dim,
                 drop_path,
                 conv_kind,
                 seqlens,
                 depth,
                 proj_bias,
                 norm_bias,
                 num_blocks,
                 init_weights,
                 downsample=None,
                 ):
        super().__init__()
        self.vilblock = nn.ModuleList([
            ViLBlockPair(
                dim=embed_dim,
                drop_path=drop_path,
                conv_kind=conv_kind,
                seqlens=seqlens,
                proj_bias=proj_bias,
                norm_bias=norm_bias,
                num_blocks=2 * depth,
                init_weights=init_weights,
            ) for i in range(depth)
            ],
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution=seqlens, dim=embed_dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.vilblock:
            x = blk(x)
        if self.downsample is not None:
            x=self.downsample(x)
        return x

class MyViLBlockDec(nn.Module):
    """ A basic Vision xLSTM layer for one stage in decoder.
    """
    def __init__(self,
                 embed_dim,
                 drop_path,
                 conv_kind,
                 seqlens,
                 depth,
                 proj_bias,
                 norm_bias,
                 num_blocks,
                 init_weights,
                 upsample=None,
                 ):
        super().__init__()
        self.vilblock = nn.ModuleList([
            ViLBlockPair(
                dim=embed_dim,
                drop_path=drop_path,
                conv_kind=conv_kind,
                seqlens=seqlens,
                proj_bias=proj_bias,
                norm_bias=norm_bias,
                num_blocks=2 * depth,
                init_weights=init_weights,
            ) for i in range(depth)
        ],
        )

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution=seqlens, dim=embed_dim,
                                       dim_scale=2,)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.vilblock:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x



class UViL(nn.Module):
    def __init__(self,
                 input_shape=(1, 224, 224),    # input shape(resolution)
                 num_classes=10,               # number of classes
                 patch_size=4,                 # size of patch
                 embed_dim=96,                 # latent dimension
                 patch_norm=True,              # whether to use patch norm
                 norm_layer=nn.LayerNorm,
                 depth=2,                      # number of ViL blocks
                 stride=None,                  # stride for patch embedding. Put None for non-overlapping patches
                 num_stages=2,                 # number of stages for U-Net like architecture
                 output_shape=None,
                 mode = "features",
                 pooling="to_image",
                 drop_path_rate=0.0,
                 drop_path_decay=False,
                 legacy_norm=False,
                 conv_kind="2d",
                 conv_kernel_size=3,
                 proj_bias=True,
                 norm_bias=True,
                 init_weights="original",
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.img_size = input_shape[-1]
        self.in_chans = input_shape[0]
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer
        self.depths = depth
        self.num_stages = num_stages
        self.stride = stride
        self.output_shape = output_shape
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.init_weights = init_weights

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=embed_dim,
            stride=stride,
            num_channels=self.in_chans,
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )

        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=embed_dim)

        # calculate stochastic depth per block
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_stages)]
        else:
            dpr = [drop_path_rate] * self.num_stages

        current_patch_resolution = self.patch_embed.seqlens
        # print(current_patch_resolution)# e.g. [H_p, W_p] or [D_p, H_p, W_p]

        # build encoder and bottleneck layers
        self.enc_layers = nn.ModuleList()
        for idx_stage in range(self.num_stages):
            layer = MyViLBlockEnc(
                embed_dim=int(self.embed_dim * 2 ** idx_stage),
                drop_path=dpr[idx_stage],
                conv_kind=conv_kind,
                seqlens=(
                    current_patch_resolution[0] // (2 ** idx_stage),
                    current_patch_resolution[1] // (2 ** idx_stage)),
                depth=depth,
                proj_bias=proj_bias,
                norm_bias=norm_bias,
                num_blocks=2 * depth,
                init_weights=init_weights,
                downsample = PatchMerging if (idx_stage < self.num_stages - 1) else None,
            )
            self.enc_layers.append(layer)

        # build decoder layers
        self.dec_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for idx_stage in range(self.num_stages):
            linear_concat = nn.Linear(in_features=2 * int(embed_dim * 2 ** (self.num_stages - 1 - idx_stage)),
                                      out_features=int(embed_dim * 2 ** (self.num_stages - 1 - idx_stage)),) if idx_stage > 0 else nn.Identity()
            if idx_stage == 0:
                dec_layer = PatchExpand(input_resolution=(current_patch_resolution[0] // (2 ** (self.num_stages -1 - idx_stage)),
                                                               current_patch_resolution[1] // (2 ** (self.num_stages - 1 - idx_stage))),
                                             dim=embed_dim * 2 ** (num_stages - 1 - idx_stage),
                                             dim_scale=2,
                                             norm_layer=norm_layer)
            else:
                dec_layer = MyViLBlockDec(
                    embed_dim=int(self.embed_dim * 2 ** (self.num_stages -1 - idx_stage)),
                    drop_path=dpr[idx_stage],
                    conv_kind=conv_kind,
                    seqlens=(
                        current_patch_resolution[0] // (2 ** (self.num_stages -1 - idx_stage)),
                        current_patch_resolution[1] // (2 ** (self.num_stages -1 - idx_stage))),
                    depth=depth,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                    num_blocks=2 * depth,
                    init_weights=init_weights,
                    upsample=PatchExpand if (idx_stage < self.num_stages - 1) else None,
                )

            self.dec_layers.append(dec_layer)
            self.concat_back_dim.append(linear_concat)

        self.norm_upsample = norm_layer(self.embed_dim)
        self.final_patch_expnad = FinalPatchExpand_X4(input_resolution=(self.img_size // self.patch_size,
                                                                self.img_size // self.patch_size),
                                              dim_scale=4, dim=embed_dim,)
        # print(f"img_size: {self.img_size}, patch_size: {self.patch_size}, dim: {embed_dim}")
        self.final_conv = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)


    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        #add positional embedding
        x = self.pos_embed(x)
        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")
        x_downsample = []

        # ViL blocks
        for idx, block in enumerate(self.enc_layers):
            x_downsample.append(x)
            x = block(x)
            # print(f"x shape for {idx} is {x.shape}")
        # x = self.norm(x)
        return x , x_downsample

    def foraward_decoder(self, x, x_downsample):
        for idx, dec_layer in enumerate(self.dec_layers):
            if idx == 0:
                x = dec_layer(x)
            else:
                # print(f"idx: {idx}")
                # print(f"x encoder shape: {x.shape}")
                # print(f"x_downsample shape for {len(x_downsample)-1-idx}: {x_downsample[len(x_downsample)-1-idx].shape}")
                # print(f"len(x_downsample): {len(x_downsample)}")
                x = torch.cat((x, x_downsample[len(x_downsample)-1-idx]), dim=-1)
                x = self.concat_back_dim[idx](x)
                x = dec_layer(x)
        self.norm_upsample(x)
        return x

    def final_4x_upsample(self, x):
        H, W = self.patch_embed.seqlens
        # print(f"H: {H}, W: {W}")
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        # print(f"x before patch expand in final: {x.shape}")
        x = self.final_patch_expnad(x)
        # print(f"x after patch expand in final: {x.shape}")
        x = x.view(B, 4 * H, 4 * W, -1)
        # print(f"x after view: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        # print(x.shape)
        x = self.final_conv(x)
        return x

    def forward(self, x):   # x: (B, C, H, W)
        x, x_downsample = self.forward_encoder(x)
        x = self.foraward_decoder(x, x_downsample)
        x = self.final_4x_upsample(x)
        return x

if __name__ == "__main__":
    model = UViL(input_shape=(1, 224, 224),
                 num_classes=4,
                 patch_size=4,
                 embed_dim=96,
                 patch_norm=True,
                 num_stages=4)
    input = torch.randn(1, 1, 224, 224)
    output = model(input)
    print(f"input shape: {input.shape} -> output shape: {output.shape}")
    summary(model, input_data=input, col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=4,
            row_settings=["var_names"])

    flops, macs, params = calculate_flops(model, input_shape=(1, 1, 224, 224))
    print("FLOPs:", flops, "MACs:", macs, "Params:", params)