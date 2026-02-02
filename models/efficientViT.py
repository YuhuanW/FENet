from typing import Dict, List, Tuple, Union, Optional, Type, Callable, Any
from inspect import signature
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = [
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
]

#################################################################################
#                             Basic Layers                                      #
#################################################################################

def build_kwargs_from_config(config: Dict, target_func: Callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

REGISTERED_NORM_DICT: Dict[str, Type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
}

def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None

REGISTERED_ACT_DICT: Dict[str, Type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
}

def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def list_sum(x: List) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])

def merge_tensor(x: List[torch.Tensor], mode="cat", dim=1) -> torch.Tensor:
    if mode == "cat":
        return torch.cat(x, dim=dim)
    elif mode == "add":
        return list_sum(x)
    else:
        raise NotImplementedError

def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")

def val2list(x: Union[List, Tuple, Any], repeat_time=1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout_rate=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Union[int, Tuple[int, int], List[int], None] = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)
    
    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class LiteMSA(nn.Module):
    r""" Lightweight multi-scale attention """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
    ):
        super(LiteMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim, bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0 : self.dim].clone(),
            multi_scale_qkv[..., self.dim : 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim :].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out


#########################################################
#                  Window Lite MSA                      #
#########################################################

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """??????????????

    Args:
        x: (B, C, H, W)
        window_size: ????

    Returns:
        windows: (num_windows * B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """?????????????

    Args:
        windows: (num_windows * B, num_channels * patch_size * patch_size, window_size, window_size)  <- ???????
                 ?? (num_windows * B, C, window_size, window_size)
        window_size: ????
        H: ????????
        W: ????????

    Returns:
        x: (B, C, H, W)
    """
    num_windows = windows.shape[0] // (H * W // window_size // window_size)
    # ????? C
    if len(windows.shape) == 4:
        C = windows.shape[1]
        x = windows.view(num_windows, H // window_size, W // window_size, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    elif len(windows.shape) == 5:
        num_features_per_window = windows.shape[1]
        x = windows.view(num_windows, H // window_size, W // window_size, window_size, window_size, num_features_per_window)
    else:
        raise ValueError(f"Unexpected shape for windows: {windows.shape}")
    return x

class WindowLiteMSA(nn.Module):
    r""" ??????????????????? """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        window_size: int = 4,  # ???????
    ):
        super().__init__()
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.total_dim = total_dim

        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim, bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

        # ??????????????? (?????????)
        self.aggreg_pool = nn.ModuleList(
            [
                nn.AvgPool2d(kernel_size=scale, stride=1, padding=get_same_padding(scale))
                for scale in scales
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = list(x.size())
        window_size = self.window_size

        # ???????
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        # ????
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        # 1. ????
        windows = window_partition(x_padded, window_size)  # (num_windows * B, C, window_size, window_size)
        num_windows_batch = windows.shape[0]

        # 2. ????? QKV (?????)
        qkv_windows = self.qkv(windows)  # (num_windows * B, 3 * total_dim, window_size, window_size)

        multi_scale_qkv_windows = [qkv_windows]
        # ???????????????
        for op in self.aggreg_pool:
            multi_scale_qkv_windows.append(op(qkv_windows))

        multi_scale_qkv_windows = torch.cat(multi_scale_qkv_windows, dim=1)  # (num_windows * B, -1, window_size, window_size)

        multi_scale_qkv_windows = torch.reshape(
            multi_scale_qkv_windows,
            (
                num_windows_batch,
                -1,
                3 * self.dim,
                window_size * window_size,
            ),
        )
        multi_scale_qkv_windows = torch.transpose(multi_scale_qkv_windows, -1, -2)
        q_windows, k_windows, v_windows = (
            multi_scale_qkv_windows[..., 0 : self.dim].clone(),
            multi_scale_qkv_windows[..., self.dim : 2 * self.dim].clone(),
            multi_scale_qkv_windows[..., 2 * self.dim :].clone(),
        )

        # 3. ??????????????
        q_windows = self.kernel_func(q_windows)
        k_windows = self.kernel_func(k_windows)

        trans_k_windows = k_windows.transpose(-1, -2)

        v_windows = F.pad(v_windows, (0, 1), mode="constant", value=1)
        kv_windows = torch.matmul(trans_k_windows, v_windows)
        out_windows = torch.matmul(q_windows, kv_windows)
        out_windows = out_windows[..., :-1] / (out_windows[..., -1:] + 1e-15)

        # 4. ????
        out_windows = torch.transpose(out_windows, -1, -2)
        out_windows = torch.reshape(out_windows, (num_windows_batch, -1, window_size, window_size))
        out_padded = window_reverse(out_windows, window_size, x_padded.shape[2], x_padded.shape[3])

        # ????
        if pad_h > 0 or pad_w > 0:
            out = out_padded[:, :, :H, :W]
        else:
            out = out_padded

        # 5. ?????
        out = self.proj(out)
        print("out shape:", out.shape)

        return out

class WindowsLite2MSA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        window_size: int = 4,
        shift_size=2,
    ):
        super().__init__()
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        self.num_groups = 4 # 可以调整分组数量

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.total_dim = total_dim

        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            groups=self.num_groups, # 使用分组卷积
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim, bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

        self.aggreg_pool = nn.ModuleList(
            [
                nn.AvgPool2d(kernel_size=scale, stride=1, padding=get_same_padding(scale))
                for scale in scales
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = list(x.size())
        window_size = self.window_size

        if self.shift_size > 0:  # 如果启用移动窗口
            shift_size = self.shift_size
            x_padded = F.pad(x, (0, (window_size - W % window_size) % window_size, 0, (window_size - H % window_size) % window_size))
            x_padded = torch.roll(x_padded, shifts=(-shift_size, -shift_size), dims=(2, 3))
        else:
            x_padded = F.pad(x, (0, (window_size - W % window_size) % window_size, 0, (window_size - H % window_size) % window_size))

        if H <= window_size and W <= window_size:
            qkv = self.qkv(x)
            multi_scale_qkv = [qkv]
            for op in self.aggreg:
                multi_scale_qkv.append(op(qkv))
            multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
            multi_scale_qkv = torch.reshape(
                multi_scale_qkv,
                (
                    B,
                    -1,
                    3 * self.dim,
                    H * W,
                ),
            )
            multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
            q, k, v = (
                multi_scale_qkv[..., 0: self.dim].clone(),
                multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
                multi_scale_qkv[..., 2 * self.dim:].clone(),
            )

            q = self.kernel_func(q)
            k = self.kernel_func(k)

            trans_k = k.transpose(-1, -2)

            v = F.pad(v, (0, 1), mode="constant", value=1)
            kv = torch.matmul(trans_k, v)
            out = torch.matmul(q, kv)
            out = out[..., :-1] / (out[..., -1:] + 1e-15)

            out = torch.transpose(out, -1, -2)
            out = torch.reshape(out, (B, -1, H, W))
            out = self.proj(out)
            return out

        else:
            windows = window_partition(x_padded, window_size)  # (num_windows * B, C, window_size, window_size)
            num_windows_batch = windows.shape[0]
            qkv_windows = self.qkv(windows)  # (num_windows * B, 3 * total_dim, window_size, window_size)

            multi_scale_qkv_windows = [qkv_windows]
            for op in self.aggreg_pool:
                multi_scale_qkv_windows.append(op(qkv_windows))

            multi_scale_qkv_windows = torch.cat(multi_scale_qkv_windows, dim=1)  # (num_windows * B, -1, window_size, window_size)

            multi_scale_qkv_windows = torch.reshape(
                multi_scale_qkv_windows,
                (
                    num_windows_batch,
                    -1,
                    3 * self.dim,
                    window_size * window_size,
                ),
            )
            multi_scale_qkv_windows = torch.transpose(multi_scale_qkv_windows, -1, -2)
            q_windows, k_windows, v_windows = (
                multi_scale_qkv_windows[..., 0 : self.dim].clone(),
                multi_scale_qkv_windows[..., self.dim : 2 * self.dim].clone(),
                multi_scale_qkv_windows[..., 2 * self.dim :].clone(),
            )

            q_windows = self.kernel_func(q_windows)
            k_windows = self.kernel_func(k_windows)

            trans_k_windows = k_windows.transpose(-1, -2)

            v_windows = F.pad(v_windows, (0, 1), mode="constant", value=1)
            kv_windows = torch.matmul(trans_k_windows, v_windows)
            out_windows = torch.matmul(q_windows, kv_windows)
            out_windows = out_windows[..., :-1] / (out_windows[..., -1:] + 1e-15)

            out_windows = torch.transpose(out_windows, -1, -2)
            out_windows = torch.reshape(out_windows, (num_windows_batch, -1, window_size, window_size))
            out_padded = window_reverse(out_windows, window_size, x_padded.shape[2], x_padded.shape[3])

            if self.shift_size > 0:  # 如果启用移动窗口，恢复偏移
                out_padded = torch.roll(out_padded, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

            out = out_padded[:, :, :H, :W]  # 移除填充
            out = self.proj(out)
            return out



class EfficientViTBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 heads_ratio: float = 1.0,
                 dim=32,
                 expand_ratio: float = 4,
                 norm="bn2d",
                 act_func="hswish",
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0.,
                 drop=0.,
                 window_size=4):
        super(EfficientViTBlock, self).__init__()
        self.window_size = window_size
        self.context_module = ResidualBlock(
           LiteMSA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
            ),
            IdentityLayer(),
        )

        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        # x = self.attn(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: Dict[str, nn.Module],
        merge_mode: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: Dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge_mode = merge_mode
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        feat = merge_tensor(feat, self.merge_mode, dim=1)
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x

class EfficientViTBackbone(nn.Module):
    def __init__(self, width_list: List[int], depth_list: List[int], in_channels=3, dim=32, expand_ratio=4, norm="bn2d", act_func="hswish") -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 224, 224))]
    @staticmethod
    def build_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float, norm: str, act_func: str, fewer_norm: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:      
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        res = []
        x = self.input_stem(x)
        res.append(x)
        for stage_id, stage in enumerate(self.stages, 1):
            x = stage(x)
            res.append(x)

        return res

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        k = k[9:]
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def efficientvit_b0(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b1(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b2(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 2, 2, 2, 2],
        # depth_list=[1, 3, 4, 4, 6],
        # act_func="relu",
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b3(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RF2B(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RF2B, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel*4, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x4 = self.branch4(self.conv(x) + x3)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))

        x = self.relu(x0 + x_cat)
        return x

if __name__ == '__main__':
    model = efficientvit_b1()
    weights = torch.load('b1-r288.pt')['state_dict']
    model.load_state_dict(update_weight(model.state_dict(), weights))
    inputs = torch.randn((1, 3, 640, 640))
    res = model(inputs)
    for i in res:
        print(i.size())
