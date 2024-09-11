
from __future__ import annotations

import torch.nn as nn
import torch

from monai.networks.blocks.selfattention import SABlock
from monai.networks.blocks.mlp import MLPBlock
from monai.utils import optional_import
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class CABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, lx, hx):
        l_output = self.input_rearrange(self.qkv(lx))
        h_output = self.input_rearrange(self.qkv(hx))
        q, k, v = l_output[0], h_output[1], h_output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            # no gradients and new tensor;
            # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        lx = q + torch.einsum("bhxy,bhyd->bhxd", att_mat, v) # q is from eq 3 in https://arxiv.org/pdf/2406.07146
        lx = self.out_rearrange(lx)
        lx = self.out_proj(lx)
        lx = self.drop_output(lx)
        return lx

class CrossAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.l_self_attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.h_self_attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.cross_attn = CABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, lx, hx):
        lx, hx = self.l_self_attn(self.norm1(lx)), self.h_self_attn(self.norm1(hx))
        lx = self.cross_attn(lx, hx)
        lx = lx + self.mlp(self.norm2(lx))
        hx = hx + self.mlp(self.norm2(hx))
        return lx, hx


class HILTProjector(nn.Module):
    def __init__(
            self,
            layer_num=2,
            proj_out_num=512,
            hidden_size=768,
            mlp_dim=3072,
            out_dim=3072,
            num_heads=12
            ):
        
        super().__init__()
        self.proj_out_num = proj_out_num
        self.out_dim = out_dim
        self.blocks = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(hidden_size, mlp_dim, num_heads)
                for i in range(layer_num)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.projector = nn.Sequential(nn.Linear(hidden_size, out_dim),
                                        nn.GELU(),
                                        nn.Linear(out_dim, out_dim))

    def forward(self, x):
        hx, lx = x
        for blk in self.blocks:
            lx, hx = blk(lx, hx)
        lx = self.projector(self.norm(lx))
        return lx