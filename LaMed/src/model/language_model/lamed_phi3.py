from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM, AutoTokenizer

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM

class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"


class LamedPhi3Model(LamedMetaModel, Phi3Model):
    config_class = LamedPhi3Config
    def __init__(self, config: Phi3Config):
        super(LamedPhi3Model, self).__init__(config)


class LamedPhi3ForCausalLM(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLM, self).__init__(config)
        self.model = LamedPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,

            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                segs
            )            
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        segs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                segs
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        output_ids = super().generate(
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("lamed_phi3", LamedPhi3Config)
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLM)