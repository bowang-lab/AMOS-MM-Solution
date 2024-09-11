from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss
import torch.nn.functional as F
from .multimodal_projector import FullLinear, HILTProjector, SpatialPoolingProjector, MLP

projector_mapping = {
    "hilt": HILTProjector,
    "spp": SpatialPoolingProjector,
    "linear": FullLinear,
    "mlp": MLP
}


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        old_patch_size = None
        if hasattr(self.config, 'patch_size'):
            old_patch_size = self.config.patch_size
        self.config.patch_size = model_args.patch_size
        self.config.any_res_crops = model_args.any_res_crops
        self.config.concat_after_project = model_args.concat_after_project
        self.config.any_res_image_size = model_args.any_res_image_size
        self.config.with_seg = model_args.with_seg
        self.config.multipler = 1

        if self.config.with_seg:
            self.config.multipler += 1

        if self.config.any_res_crops:    
            from functools import reduce
            import operator
            for res_config in self.config.any_res_crops:
                self.config.multipler += reduce(operator.mul, res_config, 1)
            print(f"AnyRes token multipler: {self.config.multipler}")
        
        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
    
        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)

        if old_patch_size != None and tuple(old_patch_size) != tuple(self.config.patch_size):
            print(f"Patch size is changed from {tuple(old_patch_size)} to {self.config.patch_size}. Creating new embedding layer")
            self.get_vision_tower().change_patch_size(self.config.patch_size)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)
        elif self.config.mm_projector_type and not isinstance(self.mm_projector, projector_mapping[self.config.mm_projector_type]):
            print(f"Creating a new project: {type(self.mm_projector)} and {projector_mapping[self.config.mm_projector_type]} does not match")
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

        self.mm_projector.multipler = self.config.multipler

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images, segs=None):
        if segs != None:
            seg_features = self.get_model().get_vision_tower()(segs)
            seg_features = self.get_model().mm_projector(seg_features)
        
        if isinstance(self.config.any_res_crops, list):
            for crop_config in self.config.any_res_crops:
                _, _, D, H, W = images.shape
                crop_depth, crop_height, crop_width = crop_config

                new_D = D // crop_depth
                new_H = H // crop_height
                new_W = W // crop_width

                embeddings = []
                
                for d in range(crop_depth): # do the crops 
                    d_start = d * new_D
                    d_end = d_start + new_D
                    for h in range(crop_height):
                        h_start = h * new_H
                        h_end = h_start + new_H
                        for w in range(crop_width):
                            w_start = w * new_W
                            w_end = w_start + new_W
                            sliced = images[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                            e = self.get_model().get_vision_tower()(sliced)
                            embeddings.append(e)
                del sliced

                # do the entire image
                images = F.interpolate(images, size=(32, 256, 256), mode='trilinear', align_corners=False) # TODO MOHD: do not hard code

                e = self.get_model().get_vision_tower()(images)
                embeddings.append(e)

                projected_embeddings = [self.get_model().mm_projector(e) for e in embeddings]
                image_features = torch.cat(projected_embeddings, dim=1)

                if segs != None:
                    image_features = torch.cat([image_features, seg_features], dim=1)

                return image_features
            
        return_all = True if self.config.mm_projector_type == "tp" else False
        image_features = self.get_model().get_vision_tower()(images, return_all)
        image_features = self.get_model().mm_projector(image_features)
        if segs != None:
            image_features = torch.cat([image_features, seg_features], dim=1)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, segs=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images(images, segs)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")