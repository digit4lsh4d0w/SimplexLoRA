# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using VeRA for sequence classification

# %% [markdown]
# In this example, we fine-tune Roberta on a sequence classification task using VeRA.

# %% [markdown]
# ## Imports

# %%
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    VeraConfig,
    PeftType,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoConfig
from tqdm import tqdm

# %% [markdown]
# ## Parameters

# %%
batch_size = 128
model_name_or_path = "roberta-base"
task = "mrpc"
peft_type = PeftType.VERA
device = "cuda"
num_epochs = 5  # for best results, increase this number
rank = 8        # for best results, increase this number
max_length = 128
torch.manual_seed(0)

# %%
peft_config = VeraConfig(
    task_type="SEQ_CLS", 
    r=rank,
    d_initial=0.1,
    target_modules=["query", "value", "intermediate.dense"],
    save_projection=True,
)
head_lr = 1e-2
vera_lr = 2e-2

# %% [markdown]
# ## Loading data

# %%
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


# %%
def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


# %%
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

# %% [markdown]
# ## Preparing the VeRA model

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %%
optimizer = AdamW(
    [
        {"params": [p for n, p in model.named_parameters() if "vera_lambda_" in n], "lr": vera_lr},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": head_lr},
    ]
)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# %% [markdown]
# ## Training

# %%
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)

# %% [markdown]
# ## Share adapters on the 🤗 Hub

# %%
account_id = ...  # your Hugging Face Hub account ID

# %%
model.push_to_hub(f"{account_id}/roberta-large-peft-vera")

# %% [markdown]
# ## Load adapters from the Hub
#
# You can also directly load adapters from the Hub using the commands below:

# %%
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer

# %%
peft_model_id = f"{account_id}/roberta-large-peft-vera"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# %%
# Load the Vera model
inference_model = PeftModel.from_pretrained(inference_model, peft_model_id)

# %%
inference_model.to(device)
inference_model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    batch.to(device)
    with torch.no_grad():
        outputs = inference_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )

eval_metric = metric.compute()
print(eval_metric)

# %%
