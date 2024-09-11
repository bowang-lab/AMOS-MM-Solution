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

from __future__ import annotations

from collections.abc import Sequence

import math
import torch
from functools import partial
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

class PatchEmbeddingBlockWrapper(PatchEmbeddingBlock):
    def __init__(self, interpolate_offset=0.1, interpolate_antialias=False, **kwargs):
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias
        self.original_patch_size = kwargs['patch_size']
        self.patch_size = kwargs['patch_size']
        self.original_image_size = kwargs['img_size']
        super().__init__(**kwargs)

    def forward(self, x):
        B, nc, d, h, w = x.shape
        x = self.patch_embeddings(x)
        if self.proj_type  == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.interpolate_pos_encoding(x, w, h, d)
        embeddings = self.dropout(embeddings)
        return embeddings

    def interpolate_pos_encoding(self, x, w, h, d): 
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.position_embeddings.shape[1] 
        if npatch == N and w == h:
            return self.position_embeddings
        pos_embed = self.position_embeddings.float()

        # return self.position_embeddings
        dim = x.shape[-1]
        d0 = d // self.patch_size[0]
        w0 = h // self.patch_size[1]
        h0 = w // self.patch_size[2]
        
        # Recover the number of patches in each dimension
        ND = self.original_image_size[0] // self.original_patch_size[0]
        NH = self.original_image_size[1] // self.original_patch_size[1]
        NW = self.original_image_size[2] // self.original_patch_size[2]

        assert N == NH * NW * ND
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / NW
            sy = float(h0 + self.interpolate_offset) / NH
            sz = float(d0 + self.interpolate_offset) / ND
            kwargs["scale_factor"] = (sz, sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (d0, w0, h0)
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, ND, NW, NH, dim).permute(0, 4, 1, 2, 3),
            mode="trilinear",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (d0, w0, h0) == pos_embed.shape[-3:]
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding_maker = partial(PatchEmbeddingBlockWrapper,
            in_channels=in_channels,
            img_size=(32, 256, 256), #TODO MOHD: to do not hard code
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.patch_embedding = self.patch_embedding_maker(patch_size=patch_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def change_patch_size(self, patch_size):
        self.patch_embedding = self.patch_embedding_maker(patch_size=patch_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        return x, hidden_states_out





class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )
    
    def feature_select(self, image_forward_outs, layers=[5,7,10,11]):
        image_feature_list = []
        for l in layers:
            image_feature_list.append(image_forward_outs.hidden_states[l])
        image_features_multi = torch.cat(image_feature_list, dim=2)

        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            image_features_multi = image_features_multi[:, 1:]

        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, image_features_multi
    
    def process_cls(self, image_features):
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, return_all=False, layers=[5,7,10,11]):
        
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        image_features = self.process_cls(image_features)
        if return_all:
            hidden_states = [hidden_states[i] for i in layers]
            hidden_states = torch.cat(hidden_states, dim=2)
            hidden_states = self.process_cls(hidden_states)
            return image_features, hidden_states
        return image_features
    
    def change_patch_size(self, patch_size):
        self.vision_tower.change_patch_size(patch_size)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size