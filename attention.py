import torch
from torch import nn, Tensor, LongTensor
from typing import Tuple, Optional
from einops import rearrange
import torch.nn.functional as F
from efficientnet_pytorch.model import MemoryEfficientSwish
from torch.nn import init
from torch.nn.parameter import Parameter

from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.helpers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d
from timm.layers.create_conv2d import create_conv2d
import math

__all__ = [
    "SimAM",
    "BiLevelRoutingAttention",
    "BiLevelRoutingAttention_nchw",
    "EfficientAttention",
    "SEAttention",
    "ECAAttention",
    "GAM_Attention",
    "GCT",
    "GatherExcite",
    "MHSA",
    "ShuffleAttention",
    "EMA"
]


# SimAM Attention Module
# START
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)


# END OF SimAm Attention Module


# BiLevelRouting Attention Module
# START
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(
        self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False
    ):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim**-0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(
            key
        )  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(
            -2, -1
        )  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(
            attn_logit, k=self.topk, dim=-1
        )  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight="none"):
        super().__init__()
        assert mul_weight in ["none", "soft", "hard"]
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(
                -1, p2, -1, -1, -1
            ),  # (n, p^2, p^2, w^2, c_kv) without mem cpy
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(
                -1, -1, -1, w2, c_kv
            ),  # (n, p^2, k, w^2, c_kv)
        )

        if self.mul_weight == "soft":
            topk_kv = (
                r_weight.view(n, p2, topk, 1, 1) * topk_kv
            )  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == "hard":
            raise NotImplementedError("differentiable hard routing TBA")
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        n_win=7,
        qk_dim=None,
        qk_scale=None,
        kv_per_win=4,
        kv_downsample_ratio=4,
        kv_downsample_kernel=None,
        kv_downsample_mode="identity",
        topk=4,
        param_attention="qkvo",
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        side_dwconv=3,
        auto_pad=True,
    ):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert (
            self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        ), "qk_dim and dim must be divisible by num_heads!"
        self.scale = qk_scale or self.qk_dim**-0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = (
            nn.Conv2d(
                dim,
                dim,
                kernel_size=side_dwconv,
                stride=1,
                padding=side_dwconv // 2,
                groups=dim,
            )
            if side_dwconv > 0
            else lambda x: torch.zeros_like(x)
        )

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (
            self.param_routing and not self.diff_routing
        )  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=self.topk,
            diff_routing=self.diff_routing,
            param_routing=self.param_routing,
        )
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = "soft"
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = "hard"
        else:  # hard non-differentiable routing
            mul_weight = "none"
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == "qkvo":
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == "qkv":
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(
                f"param_attention mode {self.param_attention} is not surpported!"
            )

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == "ada_avgpool":
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == "ada_maxpool":
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == "maxpool":
            assert self.kv_downsample_ratio is not None
            self.kv_down = (
                nn.MaxPool2d(self.kv_downsample_ratio)
                if self.kv_downsample_ratio > 1
                else nn.Identity()
            )
        elif self.kv_downsample_mode == "avgpool":
            assert self.kv_downsample_ratio is not None
            self.kv_down = (
                nn.AvgPool2d(self.kv_downsample_ratio)
                if self.kv_downsample_ratio > 1
                else nn.Identity()
            )
        elif self.kv_downsample_mode == "identity":  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == "fracpool":
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError("fracpool policy is not implemented yet!")
        elif kv_downsample_mode == "conv":
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError("conv policy is not implemented yet!")
        else:
            raise ValueError(
                f"kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!"
            )

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        x = rearrange(x, "n c h w -> n h w c")
        # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(
                x, (0, 0, pad_l, pad_r, pad_t, pad_b)  # dim=-1  # dim=-2
            )  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, "n p2 h w c -> n p2 (h w) c")
        kv_pix = self.kv_down(rearrange(kv, "n p2 h w c -> (n p2) c h w"))
        kv_pix = rearrange(
            kv_pix, "(n j i) c h w -> n (j i) (h w) c", j=self.n_win, i=self.n_win
        )

        q_win, k_win = q.mean([2, 3]), kv[..., 0 : self.qk_dim].mean(
            [2, 3]
        )  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(
            rearrange(
                kv[..., self.qk_dim :],
                "n (j i) h w c -> n c (j h) (i w)",
                j=self.n_win,
                i=self.n_win,
            ).contiguous()
        )
        lepe = rearrange(
            lepe, "n c (j h) (i w) -> n (j h) (i w) c", j=self.n_win, i=self.n_win
        )

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(
            r_idx=r_idx, r_weight=r_weight, kv=kv_pix
        )  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(
            k_pix_sel, "n p2 k w2 (m c) -> (n p2) m c (k w2)", m=self.num_heads
        )  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(
            v_pix_sel, "n p2 k w2 (m c) -> (n p2) m (k w2) c", m=self.num_heads
        )  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(
            q_pix, "n p2 w2 (m c) -> (n p2) m w2 c", m=self.num_heads
        )  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
            q_pix * self.scale
        ) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = (
            attn_weight @ v_pix_sel
        )  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(
            out,
            "(n j i) m (h w) c -> n (j h) (i w) (m c)",
            j=self.n_win,
            i=self.n_win,
            h=H // self.n_win,
            w=W // self.n_win,
        )

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")


# END OF BiLevelRouting Attention Module


# BiLevelRouting Attention NCHW Module
# START
def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    """
    Args:
        x: BCHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(
        B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1]
    )
    x = (
        torch.einsum("bmdhpwq->bmhwpqd", x).flatten(2, 3).flatten(-3, -2)
    )  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
    """
    Args:
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum("bmhwpqd->bmdhpwq", x).reshape(
        bs, nhead * head_dim, region_h * region_size[0], region_w * region_size[1]
    )
    return x


def regional_routing_attention_torch(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    region_graph: LongTensor,
    region_size: Tuple[int],
    kv_region_size: Optional[Tuple[int]] = None,
    auto_pad=True,
) -> Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if q_pad_b > 0 or q_pad_r > 0:
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if kv_pad_r > 0 or kv_pad_b > 0:
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(
        query, region_size=region_size, num_heads=nhead
    )
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # TODO: is seperate gathering slower than fused one (our old version) ?
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(
        bs, nhead, q_nregion, topk, 1, 1
    ).expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(
        key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).expand(
            -1, -1, query.size(2), -1, -1, -1
        ),
        dim=3,
        index=broadcasted_region_graph,
    )  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(
        value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).expand(
            -1, -1, query.size(2), -1, -1, -1
        ),
        dim=3,
        index=broadcasted_region_graph,
    )  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)

    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    # TODO: mask padding region
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(
        output, region_h=q_region_h, region_w=q_region_w, region_size=region_size
    )

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


class BiLevelRoutingAttention_nchw(nn.Module):
    """Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        n_win=7,
        qk_scale=None,
        topk=4,
        side_dwconv=3,
        auto_pad=False,
        attn_backend="torch",
    ):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim must be divisible by num_heads!"
        self.head_dim = self.dim // self.num_heads
        self.scale = (
            qk_scale or self.dim**-0.5
        )  # NOTE: to be consistent with old models.

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = (
            nn.Conv2d(
                dim,
                dim,
                kernel_size=side_dwconv,
                stride=1,
                padding=side_dwconv // 2,
                groups=dim,
            )
            if side_dwconv > 0
            else lambda x: torch.zeros_like(x)
        )

        ################ regional routing setting #################
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        ##########################################

        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == "torch":
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError(
                "CUDA implementation is not available yet. Please stay tuned."
            )

    def forward(self, x: Tensor, ret_attn_mask=False):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """
        N, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)

        # STEP 1: linear projection
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW

        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool2d(
            q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False
        )
        k_r = F.avg_pool2d(
            k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False
        )  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # STEP 3: token to token attention (non-parametric function)
        output, attn_mat = self.attn_fn(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            region_graph=idx_r,
            region_size=region_size,
        )

        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output


# END OF BiLevelRouting Attention NCHW Module


# Efficient Attention
# START
class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        group_split=[4, 4],
        kernel_sizes=[5],
        window_size=4,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
    ):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head**-0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(
                nn.Conv2d(
                    3 * self.dim_head * group_head,
                    3 * self.dim_head * group_head,
                    kernel_size,
                    1,
                    kernel_size // 2,
                    groups=3 * self.dim_head * group_head,
                )
            )
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(
                nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias)
            )
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(
                dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias
            )
            self.global_kv = nn.Conv2d(
                dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias
            )
            self.avgpool = (
                nn.AvgPool2d(window_size, window_size)
                if window_size != 1
                else nn.Identity()
            )

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(
        self,
        x: torch.Tensor,
        to_qkv: nn.Module,
        mixer: nn.Module,
        attn_block: nn.Module,
    ):
        """
        x: (b c h w)
        """
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = (
            mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        )  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(
        self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module
    ):
        """
        x: (b c h w)
        """
        b, c, h, w = x.size()

        q = (
            to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
        )  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = (
            to_kv(kv)
            .view(b, 2, -1, self.dim_head, (h * w) // (self.window_size**2))
            .permute(1, 0, 2, 4, 3)
            .contiguous()
        )  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        """
        x: (b c h w)
        """
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(
                self.high_fre_attntion(
                    x, self.qkvs[i], self.convs[i], self.act_blocks[i]
                )
            )
        if self.group_split[-1] != 0:
            res.append(
                self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool)
            )
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


# END of Efficient Attention


# SE Attention START
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# SE Attention END


# ECA Attention Start
class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# ECA Attention End


# GAM Attention Start


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GAM_Attention(nn.Module):
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate)
            if group
            else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate)
            if group
            else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        return out


# GAM Attention End


# GCT START
class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y**2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean**2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm**2 / 2 * self.c))
        return x * y_transform.expand_as(x)


# GCT END


# Gather And Excite Start
class GatherExcite(nn.Module):
    def __init__(
        self,
        channels,
        feat_size=None,
        extra_params=False,
        extent=0,
        use_mlp=True,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        add_maxpool=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        gate_layer="sigmoid",
    ):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert (
                    feat_size is not None
                ), "spatial feature size must be specified for global extent w/ params"
                self.gather.add_module(
                    "conv1",
                    create_conv2d(
                        channels,
                        channels,
                        kernel_size=feat_size,
                        stride=1,
                        depthwise=True,
                    ),
                )
                if norm_layer:
                    self.gather.add_module(f"norm1", nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f"conv{i + 1}",
                        create_conv2d(
                            channels, channels, kernel_size=3, stride=2, depthwise=True
                        ),
                    )
                    if norm_layer:
                        self.gather.add_module(f"norm{i + 1}", nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f"act{i + 1}", act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.mlp = (
            ConvMlp(channels, rd_channels, act_layer=act_layer)
            if use_mlp
            else nn.Identity()
        )
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x,
                    kernel_size=self.gk,
                    stride=self.gs,
                    padding=self.gk // 2,
                    count_include_pad=False,
                )
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(
                        x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2
                    )
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


# Gather And Excite End


# MHSA Start
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                requires_grad=True,
            )
            self.rel_w_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                requires_grad=True,
            )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (
                (self.rel_h_weight + self.rel_w_weight)
                .view(1, self.heads, C // self.heads, -1)
                .permute(0, 1, 3, 2)
            )  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = (
                content_position
                if (content_content.shape == content_position.shape)
                else content_position[
                    :,
                    :,
                    :c3,
                ]
            )
            assert content_content.shape == content_position.shape
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


# MHSA End

# Shuffle Attention Start
class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


# Shuffle Attention End

# EMA Start
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
# EMA End
