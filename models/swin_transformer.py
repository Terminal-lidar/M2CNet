import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape # 1xHpxWpx1
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww

        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) # qkv_bias=True
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias) # qkv_bias=True
        self.attn_drop = nn.Dropout(attn_drop) # 0
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) # 0
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, x1=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape # nW*B, window_size*window_size, C
        kv = self.kv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2,nw*B,num_heads,w*w,C//num_heads
        q = self.q_(x1).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 1,nw*B,num_heads,w*w,C//num_heads
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)     nw*B,num_heads,w*w,C//num_heads
        q = q[0]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # nw*B, num_heads, w*w, w*w
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # (nw*B, num_heads, w*w, w*w)

        if mask is not None:
            nW = mask.shape[0] # nw, w*w, w*w
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) # (nw*B, num_heads, w*w, w*w)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # nw*B,w*w,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        print(self.shift_size, self.window_size)
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(self.drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, x1):# attn_mask
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """

        B, L, C = x.shape # (B, H*W, C)
        H, W = self.H, self.W

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x1 = self.norm1(x1)
        x = x.view(B, H, W, C)
        x1 = x1.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]


        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape # (B, Hp, Wp, C)

        # cyclic shift
        if self.shift_size[0] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_x1 = x1
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C
        x1_windows = x1_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, x1=x1_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C) # nW*B, window_size0, window_size1, C
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 minpatch=False,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size # [4, 64]
        self.shift_size = [int(i//2) for i in window_size] # [2, 32]
        self.depth = depth # 4
        self.use_checkpoint = use_checkpoint # False

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, # 512, 256, 128, 64
                num_heads=num_heads,# 
                window_size=window_size,
                shift_size=[0, 0] if (i % 2 == 0) or minpatch else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, # 0
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, H, W, x1):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size) # nW, window_size[0], window_size[1], 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask, x1)
        return x, H, W

class SwinTransformer(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 4, 4, 4],
                 window_size=[4, 4],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            layer = BasicLayer(dim=int(8 * embed_dim / 2**i_layer), depth=depths[i_layer],
                num_heads=num_heads[i_layer], window_size=window_size, minpatch=False,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers-1)][::-1] # [256, 128, 64]
        self.num_features = num_features


        self.redduce1 = nn.Linear(10*embed_dim, 4*embed_dim, bias=False)
        self.redduce2 = nn.Linear(5*embed_dim, 2*embed_dim, bias=False)
        self.redduce3 = nn.Linear(int(2.5*embed_dim), embed_dim, bias=False)

        self.norm512 = nn.LayerNorm(8*embed_dim) # 512
        self.norm256 = nn.LayerNorm(4*embed_dim) # 256
        self.norm128 = nn.LayerNorm(2*embed_dim) # 128
        self.norm64 = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, out):
        """Forward function."""
        Wh, Ww = x.size(2), x.size(3) # b x 512 x 4 x 64
        x = x.flatten(2).transpose(1, 2) # b x (4x64=256) x 512 
        x = self.norm512(x)
        x, Wh, Ww = self.layers[0](x, Wh, Ww) # x:b x 256 x 512

        x = x.view(-1, Wh, Ww, 8*self.embed_dim).permute(0, 3, 1, 2).contiguous() # torch.Size([b, 512, 4, 64])

        x = nn.PixelShuffle(2)(x) # b x 128 x 8 x 128
        x = torch.cat([x, out[2]], dim=1).flatten(2).transpose(1, 2) # b x 640 x 8 x 128 -> b x (8x128) x 640
        x = self.redduce1(x) # b x (8 x 128) x 256
        # print(torch.max(x), torch.min(x))
        x = torch.clamp(x, min=-65504, max=65504) # 2^16=65536?
        x = self.norm256(x) # b x (8 x 128) x 256


        x, Wh, Ww = self.layers[1](x, 2*Wh, 2*Ww)
        x = x.view(-1, Wh, Ww, 4*self.embed_dim).permute(0, 3, 1, 2).contiguous() # b x 256 x 8 x 128

        x = nn.PixelShuffle(2)(x) # b x 64 x 16 x 256
        x = torch.cat([x, out[1]], dim=1).flatten(2).transpose(1, 2)  # b x (16 x 256) x 320 
        x = self.redduce2(x)
        x = self.norm128(x) # b x (16 x 256) x 128
        x, Wh, Ww = self.layers[2](x, 2*Wh, 2*Ww) # b x (16 x 256) x 128

        x = x.view(-1, Wh, Ww, 2*self.embed_dim).permute(0, 3, 1, 2).contiguous() # b x 128 x 16 x 256

        x = nn.PixelShuffle(2)(x) # b x 32 x 32 x 512
        x = torch.cat([x, out[0]], dim=1).flatten(2).transpose(1, 2) # b x (32x512) x 160 
        x = self.redduce3(x)
        x = self.norm64(x)
        x, Wh, Ww = self.layers[3](x, 2*Wh, 2*Ww) # x : b x (32 x 512) x 64

        return x