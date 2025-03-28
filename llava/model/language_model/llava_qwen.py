#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..multimodal_encoder.builder import build_vision_tower
from ..multimodal_projector.builder import build_vision_projector
from ..llava_arch import LlavaMetaModel
from llava.constants import (IGNORE_INDEX, 
                             IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
                            #  DEFAULT_AUDIO_TOKEN,
                             DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN,
                             DEFAULT_IM_START_INDEX, DEFAULT_IM_END_INDEX
                             )



class LlavaQwen2Config(Qwen2Config):
    model_type = "llava_Qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwen2Config

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = LlavaQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
        
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    

    def get_model(self):
        return self.model
    
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
        ):  
            vision_tower = self.get_vision_tower()
            
            if not vision_tower or images is None or input_ids.shape[1] == 1:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)


            
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == DEFAULT_IM_START_INDEX).sum() == 0:
                    
                    cur_input_embeds = cur_input_embeds + (0. * cur_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                
                if (cur_input_ids == DEFAULT_IM_START_INDEX).sum() != (cur_input_ids == DEFAULT_IM_END_INDEX).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                
                image_start_token_pos = torch.where(cur_input_ids == DEFAULT_IM_START_INDEX)[0][0]

                cur_image_features = cur_image_features.to(device=cur_input_embeds.device)
                num_image_tokens = cur_image_features.shape[0]
                if cur_input_ids[image_start_token_pos + num_image_tokens + 1] != DEFAULT_IM_END_INDEX:
                    raise ValueError("The image end token should follow the image start token.")
                new_input_embeds_i = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_image_tokens + 1:]
                            ), 
                            dim=0
                        )
                assert cur_input_embeds.size() == new_input_embeds_i.size(), "new embedding size != previous embedding size"
                new_input_embeds.append(new_input_embeds_i)
            
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            if labels != None:
                assert inputs_embeds.size()[0] == labels.size()[0], f"inputs_embeds.size()[0]={inputs_embeds.size()[0]} += labels.size()[0] = {labels.size()[0]}"
            
            
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            if labels is None:
                labels = torch.full_like(input_ids, -100)

            return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
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
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
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
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_Qwen2", LlavaQwen2Config)
AutoModelForCausalLM.register(LlavaQwen2Config, LlavaQwen2ForCausalLM)
