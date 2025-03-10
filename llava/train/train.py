#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import gzip
import torch
import pandas as pd
import transformers
import tarfile
import tokenizers
import torchaudio
from llava.constants import (IGNORE_INDEX, 
                             IMAGE_TOKEN_INDEX, 
                             DEFAULT_IMAGE_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
                            #  DEFAULT_AUDIO_TOKEN,
                             DEFAULT_AUDIO_START_TOKEN,
                             DEFAULT_AUDIO_END_TOKEN
                             )
import torch
import torch.distributed as dist
from itertools import chain
from datasets import load_dataset, interleave_datasets, concatenate_datasets
import sys
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM, Qwen2Tokenizer
from llava.train.llava_trainer import LLaVATrainer
from transformers import Trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("./WavTokenizer"))))
from llava import conversation as conversation_lib
from llava.model import *
import io
from PIL import Image
import random

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    


@dataclass
class DataArguments:
    train_metadata_names: Optional[str] = field(default=None, metadata={"help": "Comma-separated list of training data names."})
    metadata_folder: Optional[str] = field(default=None, metadata={"help": "Path to the HF training data."})
    data_path: Optional[str] = field(default=None, metadata={"help": "Path to the HF training data."})
    val_data_file_names: Optional[str] = field(default=None, metadata={"help": "Comma-separated list of validation data names."})
    val_meta_json_path: Optional[str] = field(default=None, metadata={"help": "Path to the HF validation data."})
    lazy_preprocess: bool = field(default=False, metadata={"help": "Enable lazy preprocessing."})
    is_multimodal: bool = field(default=False, metadata={"help": "Enable multimodal data processing."})
    audio_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder with audio files."})
    image_aspect_ratio: str = 'square'
    num_image_tokens: int = field(default=256, metadata={"help": "number of image tokens."})
    image_folder: Optional[str] = field(default=None, metadata={"help": "Image path"})
    text_output_ins: bool = field(default=False, metadata={"help": "Text Instruct output."})
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

def generate_new_path(old_path, prefix="COCO_train2014_"):
    directory, old_filename = os.path.split(old_path)
    new_filename = f"{prefix}{old_filename}"
    return os.path.join(directory, new_filename)


def vqav2_read_image(image, image_dir):
    image_path = os.path.join(image_dir, image)
    
    return Image.open(generate_new_path(image_path))

def inspect_model_trainability(model, output_file="model_trainability_analysis.txt"):
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    
    if rank != 0:
        return
    from contextlib import redirect_stdout
    from datetime import datetime

    with open(output_file, 'w') as f:
        with redirect_stdout(f):
            # Print header information
            print("Model Trainability Analysis Report")
            print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)

            # Analyze modules for trainability
            trainable_modules = []
            frozen_modules = []

            for name, module in model.named_modules():
                params = list(module.parameters(recurse=False))
                if params:
                    # Check if any parameter in the module has requires_grad=True
                    is_trainable = any(param.requires_grad for param in params)
                    if is_trainable:
                        trainable_modules.append(name)
                    else:
                        frozen_modules.append(name)

            # Output detailed trainability information
            print("\nTrainable Modules:")
            print("-" * 50)
            for module_name in trainable_modules:
                print(f"Module: {module_name} - Trainable")

            print("\nFrozen Modules:")
            print("-" * 50)
            for module_name in frozen_modules:
                print(f"Module: {module_name} - Frozen")

            # Summary
            print("\nSummary:")
            print("-" * 50)
            print(f"Total trainable modules: {len(trainable_modules)}")
            print(f"Total frozen modules: {len(frozen_modules)}")
    print(f"Trainability analysis has been written to {output_file}")
    
def inspect_model_dtypes(model, output_file="model_dtypes_analysis.txt"):
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    
    if rank != 0:
        return
    import sys
    from contextlib import redirect_stdout

    with open(output_file, 'w') as f:
        with redirect_stdout(f):
            # Print timestamp and model info
            from datetime import datetime
            print(f"Model Analysis Report")
            print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)

            # Initialize lists to store modules by dtype
            bf16_modules = []
            fp16_modules = []
            fp32_modules = []
            other_modules = []

            # Analyze modules
            for name, module in model.named_modules():
                params = list(module.parameters(recurse=False))
                if params:
                    # Get dtype of first parameter as reference
                    dtype = next(module.parameters()).dtype
                    module_info = {
                        'name': name,
                        'params': [(p_name, p.shape) for p_name, p in module.named_parameters(recurse=False)]
                    }
                    
                    if dtype == torch.bfloat16:
                        bf16_modules.append(module_info)
                    elif dtype == torch.float16:
                        fp16_modules.append(module_info)
                    elif dtype == torch.float32:
                        fp32_modules.append(module_info)
                    else:
                        other_modules.append((module_info, dtype))

            # Write detailed information
            print("\nModules with bfloat16:")
            print("-" * 50)
            for module in bf16_modules:
                print(f"\nModule: {module['name']}")
                for param_name, shape in module['params']:
                    print(f"  Parameter: {param_name}")
                    print(f"  Shape: {shape}")

            print("\nModules with float16:")
            print("-" * 50)
            for module in fp16_modules:
                print(f"\nModule: {module['name']}")
                for param_name, shape in module['params']:
                    print(f"  Parameter: {param_name}")
                    print(f"  Shape: {shape}")

            print("\nModules with float32:")
            print("-" * 50)
            for module in fp32_modules:
                print(f"\nModule: {module['name']}")
                for param_name, shape in module['params']:
                    print(f"  Parameter: {param_name}")
                    print(f"  Shape: {shape}")

            if other_modules:
                print("\nModules with other dtypes:")
                print("-" * 50)
                for module_info, dtype in other_modules:
                    print(f"\nModule: {module_info['name']} (dtype: {dtype})")
                    for param_name, shape in module_info['params']:
                        print(f"  Parameter: {param_name}")
                        print(f"  Shape: {shape}")

            # Write summary
            print("\nSummary:")
            print("-" * 50)
            print(f"Total bfloat16 modules: {len(bf16_modules)}")
            print(f"Total float16 modules: {len(fp16_modules)}")
            print(f"Total float32 modules: {len(fp32_modules)}")
            print(f"Total other dtype modules: {len(other_modules)}")
    print(f"Analysis has been written to {output_file}")


def safe_save_model_for_hf_trainer(trainer, ):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()
        
        
        
        
def safe_save_model_for_hf_trainer(trainer, output_dir):
    
    torch.cuda.synchronize()
    trainer.save_model(output_dir)    
        
def preprocess_stage_2(
    conversation: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
) -> Dict:
    assert len(conversation) == 2
    messages = [
        {"role": "system", "content": "You are a helpful speech-text-vision assistant."}
    ]
    
    user_prompt = conversation[0]
    
    # add placeholder for image
    if "<image>" in user_prompt["content"]:
        user_prompt["content"] = user_prompt["content"].replace("<image>", f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*data_args.num_image_tokens}{DEFAULT_IM_END_TOKEN}")
    messages.append(user_prompt)
    assistant_prompt = conversation[1]
    
    #add output ins
    if data_args.text_output_ins:
        if assistant_prompt['task'] == "tts":
            assistant_prompt["content"] = "This is how your text sounds when converted to speech: " + assistant_prompt["content"] + "."
            
        elif  assistant_prompt['task'] == "asr":
            assistant_prompt["content"] = "The transcript of the provided audio is: " + assistant_prompt["content"] + "."
            
        elif assistant_prompt['task'] in ["vqa_ttt", "vqa_stt"]:
            assistant_prompt["content"] = "Answer: " + assistant_prompt["content"] + "."
            
        elif assistant_prompt['task'] in ["vqa_tts", "vqa_sts"]:
            assistant_prompt["content"] = f"The textual answer is {assistant_prompt['text_answer']}. Therefore, the audio answer is: " + assistant_prompt["content"] + "."

        elif assistant_prompt['task'] in ["caption_ttt", "caption_stt"]:
            assistant_prompt["content"] = f"Caption: " + assistant_prompt["content"]
        elif assistant_prompt['task'] in ["caption_tts", "caption_sts"]:
            assistant_prompt["content"] = f"The textual caption is {assistant_prompt['text_answer']}. Therefore, the audio caption is: " + assistant_prompt["content"] + "."
        else:
            raise ValueError(f"you gave a wrong task {assistant_prompt['task']}")
    
    input_content = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False)
    output_content = tokenizer.apply_chat_template(
        [assistant_prompt],
        tokenize=False,
        add_generation_prompt=False)
    
    
    
    output_content = output_content.replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "").strip() + "<|endoftext|>"
    # print(input_content)
    # print(output_content)
    # print("===========================")
    input_prompt = tokenizer(input_content, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    output_prompt = tokenizer(output_content, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    input_ids = torch.concat([input_prompt, output_prompt], dim=-1)
    label_ids =torch.concat([torch.full((input_prompt.size(-1),), IGNORE_INDEX), output_prompt], dim=-1)
    assert len(label_ids) == len(label_ids), "len(label_ids) != len(label_ids)"
    return dict(input_ids=input_ids, labels=label_ids)
        
        

def preprocess(
    source: str,
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
    
) -> Dict:
    conversation = source["conversations"]
    return preprocess_stage_2(conversation, tokenizer,data_args)
    # input_prompt = tokenizer(source["question"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    # output_prompt = tokenizer(source["answer"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    # input_ids = torch.concat([input_prompt, output_prompt], dim=-1)
    # label_ids =torch.concat([torch.full((input_prompt.size(-1),), IGNORE_INDEX), output_prompt], dim=-1)
    # assert len(label_ids) == len(label_ids), "len(label_ids) != len(label_ids)"
    # return dict(input_ids=input_ids, labels=label_ids)



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, metadata_folder: str,
                 data_names: str,
                 data_path,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        data_names = data_names.split(",")
        self.list_data_dict = []
        for sub_dataset_name in data_names:
            sub_json_path = os.path.join(metadata_folder, sub_dataset_name)
            with open(sub_json_path) as f:
                sub_dataset = json.load(f)
            self.list_data_dict.extend(sub_dataset)
        
        # self.list_data_dict = [item for item in self.list_data_dict if "ttt" in item["task"]]------
        # self.list_data_dict = self.list_data_dict[:8]
        # self.list_data_dict = self.list_data_dict
        random.shuffle(self.list_data_dict)
        # self.list_data_dict = self.list_data_dict[:64]
        # with open("debug.json", "w") as f:
        #     json.dump(self.list_data_dict, f)
        self.image_processor = data_args.image_processor

        self.tokenizer = tokenizer

        self.data_args = data_args
        self.data_path = data_path
        
        # self.list_data_dict = self.list_data_dict[:128]
        # with open("checking_debug.json", "w") as f:
        #     json.dump(self.list_data_dict, f)
        rank0_print("X"*60)
        rank0_print(f"There are {len(self.list_data_dict)} examples!")
        rank0_print("X"*60)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        current_idx = i
        while current_idx < len(self.list_data_dict):  # Ensure we stay within bounds
            try:
                metadata = self.list_data_dict[current_idx]
                with open(os.path.join(self.data_path, metadata["file"])) as f:
                    data = json.load(f)
                sources = data[str(metadata["key"])]
                dataset = metadata["dataset"]
                del data

                # Handle different datasets
                if dataset in ["vqav2", "vg", "laion"]:
                    if dataset == "vqav2" or dataset == "vg":
                        image_path = os.path.join(self.data_args.image_folder, sources["image"])
                        image = Image.open(image_path).convert('RGB')
                    elif dataset == "laion":
                        tar_path = os.path.join(self.data_args.image_folder, "laion/tars/images/", sources["tar"])
                        with tarfile.open(tar_path, "r:gz") as tar:
                            image_name = sources["image"]
                            img_member = tar.getmember(image_name)
                            img_file = tar.extractfile(img_member)
                            image_data = img_file.read()
                            image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    else:
                        raise RuntimeError(f"Dataset '{dataset}' not implemented.")

                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    crop_size = self.data_args.image_processor.crop_size
                    image = torch.zeros(3, crop_size['height'], crop_size['width'])

                # Preprocess the data and construct the item
                data_dict = preprocess(sources, self.tokenizer, self.data_args)
                item = dict(
                    input_ids=data_dict["input_ids"],
                    labels=data_dict["labels"],
                    image=image
                )
                return item
            except Exception as e:
                print(f"Error processing item {current_idx}: {e}. Trying next index...")
                current_idx += 1  # Move to the next index
        raise IndexError(f"Unable to find a valid item starting from index {i}.")
            
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance for instance in instances if instance != None]
        if len(instances) == 0:
            return None
        input_ids, labels, images = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels", "image"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images = torch.stack(images)
        )   
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        data_names=data_args.train_metadata_names,
        metadata_folder=data_args.metadata_folder,
        data_path= data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args)
    # eval_dataset = LazySupervisedDataset(
    #     data_path=data_args.val_meta_json_path,
    #     data_names=data_args.val_data_file_names,
    #     split="validation",
    #     tokenizer=tokenizer,
    #     data_args=data_args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                # eval_dataset=eval_dataset,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    rank0_print(f"Model Args:\n{model_args}")
    rank0_print(f"Data Args:\n{data_args}")
    rank0_print(f"Training Args:\n{training_args}")
    
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model = LlavaQwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        ignore_mismatched_sizes=True,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        
    )

    model.config.use_cache = False
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # ignore_mismatched_sizes=True,
        # model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        )
    #Add audio tokens and special tokens 
    audio_token_size = 1024
        
    additional_tokens = []
    
    for audio_idx in range(audio_token_size):
        token = f"<||audio-{audio_idx}||>"
        if token in tokenizer.get_vocab():
            continue
            rank0_print(f"Token {token} already exists in the vocabulary.")
        else:
            additional_tokens.append(token)

    if additional_tokens:
        tokenizer.add_tokens(additional_tokens)
        rank0_print(f"Added {len(additional_tokens)} audio tokens to the tokenizer.")

    additional_special_tokens = [
        DEFAULT_AUDIO_START_TOKEN,  # Ensure DEFAULT_AUDIO_START_TOKEN is defined
        DEFAULT_AUDIO_END_TOKEN
    ]

    special_tokens = {
        'additional_special_tokens': additional_special_tokens,
        'pad_token': "[PAD]"
    }
    
    
    # Resize embedding layer
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    
    # init model
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
    
        
    # rank0_print(model)
    inspect_model_trainability(model)
    inspect_model_dtypes(model)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("LOAD MODEL§")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    


if __name__ == "__main__":
    train()
