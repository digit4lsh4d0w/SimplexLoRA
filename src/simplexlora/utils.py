import random
import torch
import peft
import os
import torch.nn as nn
import numpy as np
from typing import Callable

def get_run_name(args, parser, tuning=False):
    key_args = ["optimizer", "model", "dataset", "ft_strategy"]
    ignore_args = [
        "verbose", "seed", "run_prefix", "wandb_project", "results_path",
        "tune_path", "data_path", "wandb", "tune", "use_old_tune_params",
        "eval_runs", "augment", "run_name",
    ]
    ignore_args_tuning = [
        "n_epoches", "lr", "weight_decay", "rotate", "scale_bound",
        "weight_init", "ns_steps",
    ]
    if getattr(args, 'optimizer', None) in ["taia", "adam-sania"]:
        ignore_args_tuning.append("scale")
    
    defaults = vars(parser.parse_args([]))

    # Helper to abbreviate parameter names and values
    def abbreviate(key, value):
        mapping = {
            "batch_size": "bs",
            "learning_rate": "lr",
            "weight_decay": "wd",
            "weight_decay_w": "wdw",
            "learning_rate_w": "lrw",
            "max_fat_steps": "mfs",
            "fat_step": "fs",
            "max_seq_length": "msl",
            "max_eval_samples": "mes",
            "gradient_accumulation_steps": "gas",
            "max_train_steps": "mts",
            "lora_r": "r",
            "lora_alpha": "al",
            "lora_dropout": "drop",
            "quantization_bit": "qb",
            "eval_steps": "es",
        }
        short_key = mapping.get(key, key)
        if isinstance(value, float):
            # Use scientific notation without excessive zeros
            if value == 0.0:
                val_str = "0"
            elif abs(value) >= 1e-3:
                val_str = f"{value:.4g}".rstrip('0').rstrip('.')
            else:
                val_str = f"{value:.1e}".replace('-', 'm').replace('+', '')
        else:
            val_str = str(value)
        return f"{short_key}{val_str}"

    if not tuning:
        print("~~~~~~~~~~~~~~~ KEY ARGUMENTS ~~~~~~~~~~~~~~~")
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is None:
                value = "None"
            if key == "model":
                # Replace dangerous characters
                value = str(value).replace("/", "-").replace("\\", "-")
            if not tuning:
                print(f"{key:<20} {value}")
            prefix_parts.append(str(value))

    prefix = "_".join(prefix_parts)

    if not tuning:
        print("~~~~~~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~~~~~~")
    non_default_parts = []
    for key, value in vars(args).items():
        if not tuning:
            print(f"{key:<40} {value}")
        if key in ignore_args:
            continue
        if key in ignore_args_tuning and tuning:
            continue
        if key in key_args:
            continue
        default_val = defaults.get(key, None)
        if default_val is None or value != default_val:
            non_default_parts.append(abbreviate(key, value))

    non_default_string = "__".join(non_default_parts)

    if getattr(args, "run_prefix", None) is not None and not tuning:
        prefix = args.run_prefix + "__" + prefix

    full_name = f"{prefix}__{non_default_string}" if non_default_string else prefix

    # Optional: enforce max length (e.g., 200 chars)
    if len(full_name) > 200:
        full_name = full_name[:200].rstrip("_-")

    return full_name

def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)

def set_global_seed(seed=18):
    def seed_worker(worker_seed):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return g, seed_worker

def print_trainable_params(model, verbose=True):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for param in model.buffers():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    train_proportion = 100 * trainable_params / all_param

    if verbose:
        print(
            f"trainable params: {trainable_params / 1_000_000:.2f}M || all params: {all_param / 1_000_000_000:.2f}B || trainable%: {train_proportion:.2f}%"
        )

    return all_param, trainable_params, 100 * trainable_params / all_param

def print_trainable_parameters(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for param in model.buffers():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if verbose:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    return all_param, trainable_params

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_device(device_no: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device

def count_atapters(model, peft_type):
    if peft_type in ["LoRA", "ADALoRA", "DoRA", "rsLoRA", "WeightLoRA", "SimplexLoRA"]:
        adapter_name = "lora_A"
    elif peft_type == "LoKR":
        adapter_name = "lokr_w1"
    elif peft_type == "LoHA":
        adapter_name = "hada_w1_a" 
    elif peft_type == "VERA":
        adapter_name = "vera_lambda_b"
    elif peft_type in ["WeightLoRA", "RandLoRA"]:
        adapter_name = "weight_lora_A"
    elif peft_type == "Full":
        adapter_name = None
    else:
        raise ValueError(f"Wrong peft_type: {peft_type}")
    
    num_adapters = None
    if adapter_name is not None:
        num_adapters = 0
        for name, param in model.named_parameters():
            if adapter_name in name and param.requires_grad:
                num_adapters += 1
    
    return num_adapters

def apply_rand_weight_lora(model, n, k):
    idxs = torch.randperm(n)[:k] 
    i = 0
    for name, param in model.named_parameters():
        if "weight_lora_w" in name:
            param.requires_grad = False
            if i not in idxs:
                param.data = torch.tensor([0.])
            i += 1

def get_peft_arguments(training_args):
    if training_args.ft_strategy == "LoRA":
        peft_args = peft.LoraConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout,
            bias                = "none",
        )
    elif training_args.ft_strategy == "LoKR":
        peft_args = peft.LoKrConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "LoHA":
        peft_args = peft.LoHaConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "VERA":
        peft_args = peft.VeraConfig(
            r                   = training_args.lora_r,
            vera_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "ADALoRA":
        peft_args = peft.AdaLoraConfig(
            target_r            = training_args.lora_r,
        )
    elif training_args.ft_strategy == "DoRA":
        peft_args = peft.LoraConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout,
            use_dora            = True,
        )
    elif training_args.ft_strategy == "rsLoRA":
        peft_args = peft.LoraConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout,
            use_rslora          = True,
        )
    elif training_args.ft_strategy in ["WeightLoRA", "SimplexLoRA"]:
        peft_args = peft.LoraConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout,
            use_weight_lora     = True,
        )
    elif training_args.ft_strategy == "Full":
        return None
    else:
        raise ValueError(f"Incorrect FT type {training_args.ft_strategy}!")

    if "deberta" in training_args.model.lower():
        peft_args.target_modules = [
            "query_proj",
            "key_proj",
            "value_proj",
            "intermediate.dense",
            "output.dense",
        ]
    elif "bart" in training_args.model.lower():
        peft_args.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        ]
    elif "qwen" in training_args.model.lower():
        peft_args.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    elif "distilbert-base" in training_args.model.lower():
        peft_args.target_modules = [
            "q_lin",
            "k_lin",
            "v_lin",
        ]
    elif "flan" in training_args.model.lower() or "t5" in training_args.model.lower():
        peft_args.target_modules = [
            "v",
            "o",
            "q",
            "o",
            "wi",
        ]
    else:
        raise ValueError(f"Pass target_modules to your model {training_args.model_name}")
    return peft_args

class AdapterLayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter. Wraps an existing OPT linear layer"""
    def __init__(self, module: nn.Linear, r: int = 8, add_weight=True):
        super().__init__()
        self.module = module  # pre-trained (frozen) linear layer
        self.lora_A = nn.Linear(in_features=module.in_features,
                                out_features=r, bias=False,
                                dtype=module.weight.dtype,
                                device=module.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        self.lora_B = nn.Linear(in_features=r,
                                out_features=module.out_features, bias=False,
                                dtype=module.weight.dtype,
                                device=module.weight.device)
        nn.init.zeros_(self.lora_B.weight)
        self.add_weight = add_weight
        if add_weight:
            self.w = torch.tensor(1., requires_grad=True)

    def forward(self, x):
        frwd_module = self.module(x)
        if self.add_weight:
            frwd_adapter = self.w * self.lora_B(self.lora_A(x))
        else:
            frwd_adapter = self.lora_B(self.lora_A(x))
        return frwd_module + frwd_adapter
    
class IdOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(IdOptimizer, self).__init__(params, defaults)
        self.grad_0 = None

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.grad_0 is None:
                    self.grad_0 = p.grad.data
                else:
                    self.grad_0 += p.grad.data
        return loss
