import os
import logging
from typing import Optional, List, Dict, Union
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import AMOSCapDataset, AMOSVQADataset, UniDatasets
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM, LamedGemmaForCausalLM, LamedQwen2ForCausalLM, LamedMistralForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
from utils import parse_custom_tuple, print_trainable_parameters, process_crops

torch.cuda.set_device(0)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    use_hilt: bool = field(default=False)
    image_size: Optional[Union[tuple, str]] = field(default=(32, 256, 256))
    patch_size: Optional[Union[tuple, str]] = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)
    with_seg: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # AnyRes
    any_res_crops: Optional[str] = field(default=None)
    any_res_image_size: Optional[Union[tuple, str]] = field(default=None)
    concat_after_project: bool = field(default=True)

    # MRG
    prompt: str = field(default="")
    triplet: bool = field(default=False)

    # VQA
    only_letter: bool = field(default=False)
    with_reason: bool = field(default=False)
    with_added_q: bool = field(default=False)
    with_report: bool = field(default=False)

    splitted_findings: bool = field(default=False)
    with_template: bool = field(default=False)
    organs: List[str] = field(default_factory=lambda: ['abdomen', 'chest', 'pelvis'])
    with_impressions: bool = field(default=False)
    zoom_in: bool = field(default=False)

    def __post_init__(self):
        # Ensure image_size is either a tuple, a string, or None
        if not isinstance(self.image_size, (tuple, str)):
            raise TypeError(f"image_size must be a tuple, a string not {type(self.image_size).__name__}")

@dataclass
class DataArguments:
    json_path: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more paths to JSON files (e.g. train/CT-AMOS-Tr.json train/CT-RATE-Tr.json)"}
    )
    data_root: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more data-root directories corresponding to each JSON"}
    )
    task: str = "mrg"
    with_gen: bool = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    keep_lora: bool = False
    freeze_llm: bool = False

    # other peft
    use_dora: bool = field(default=False)

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-phi-4b-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


@dataclass
class DataCollator:
    def __call__(self, batch: list) -> dict:
        images, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask'))

        images = torch.cat([img.unsqueeze(0) for img in images if img is not None], dim=0) if images[0] is not None else None
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return return_dict


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for attr in ["image_size", "any_res_image_size", "patch_size"]:
        value = getattr(model_args, attr, None)
        if isinstance(value, str): setattr(model_args, attr, parse_custom_tuple(value))

    if isinstance(model_args.any_res_crops, str): model_args.any_res_crops = process_crops(model_args.any_res_crops)

    print('model_args:', model_args, '\n', '=='*10)
    print('data_args:', data_args, '\n', '=='*10)
    print('training_args:', training_args, '\n', '=='*10)

    local_rank = training_args.local_rank

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.pad_token == None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)
    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            print("Using llama")
            model = LamedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
        elif "gemma" in model_args.model_type:
            print("Using gemma")
            model = LamedGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
        )
        elif 'phi3' in model_args.model_type:
            print("Using phi")
            model = LamedPhi3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True
                )
        elif 'qwen' in model_args.model_type:
            print("Using qwen")
            model = LamedQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True
            )
        elif 'mistral' in model_args.model_type:
            print("Using Mistral")
            model = LamedMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.requires_grad_(True)
    model.config.use_cache = False

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    if model_args.freeze_backbone:
        model.get_vision_tower().requires_grad_(False)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    if training_args.lora_enable or training_args.keep_lora or training_args.freeze_llm:
        if training_args.freeze_llm:
            rank0_print("Freezing LLM finetuning everything else.")
            training_args.lora_enable = False
            model.requires_grad_(False)
        elif training_args.keep_lora:
            rank0_print("Keeping LoRA weights and finetuning them")
        elif training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                use_dora=training_args.use_dora,
                task_type="CAUSAL_LM",
            )
            rank0_print("Adding LoRA adapters only on LLM.")
            model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head']]
            ):
                p.requires_grad = True

    print_trainable_parameters(model)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    data_args.with_impressions = model_args.with_impressions
    data_args.prompt = model_args.prompt
    data_args.with_template = model_args.with_template

    # MRG Args
    data_args.triplet = model_args.triplet

    # VQA Args
    data_args.only_letter = model_args.only_letter

    if model_args.any_res_image_size:
       data_args.data_img_size = model_args.any_res_image_size
    else:
        data_args.data_img_size = model_args.image_size
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    eval_dataset = None
    if data_args.task == "mrg":
        train_dataset = AMOSCapDataset(data_args, tokenizer, mode='train')
    elif data_args.task == "vqa":
        train_dataset = AMOSVQADataset(data_args, tokenizer, mode='train')
    elif data_args.task == "all":
        train_dataset = UniDatasets(data_args, tokenizer=tokenizer, mode='train')
    else:
        ValueError("Please choose an approriate task.")
    
    data_collator = DataCollator()

    rank0_print("="*20 + " Training " + "="*20)
    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics
                      )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    model.model.config.architectures = model.__class__.__name__
    model._name_or_path = training_args.output_dir

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable and not training_args.keep_lora:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
        print("Merge weights with LoRA")
        model = model.merge_and_unload()
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
