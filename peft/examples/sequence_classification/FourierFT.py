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
# # Using FourierFT for sequence classification

# %% [markdown]
# In this example, we fine-tune Roberta (base) on a sequence classification task using FourierFT.

# %% [markdown]
# ## Imports

# %%
#  To run this notebook, please run `pip install evaluate` to install additional dependencies not covered by PEFT.
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    FourierFTConfig,
    PeftType,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoConfig
from tqdm import tqdm

# %% [markdown]
# ## Parameters

# %%
batch_size = 32
model_name_or_path = "roberta-base"
task = "mrpc"
peft_type = PeftType.FOURIERFT
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 5  # for better results, increase this number
n_frequency = 1000        # for better results, increase this number
scaling = 150.0
max_length = 512
torch.manual_seed(0)

# %%
peft_config = FourierFTConfig(
    task_type="SEQ_CLS", 
    n_frequency=n_frequency,
    target_modules=["query", "value"],
    scaling = scaling,
)
head_lr = 6e-3  # the learning rate for the classification head for NLU tasks
fft_lr = 6e-2   # the learning rate for the parameters other than the classification head (q,v in this case)

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
# ## Preparing the FourierFT model

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %%
head_param = list(map(id, model.classifier.parameters()))

others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 

optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": head_lr},
    {"params": others_param, "lr": fft_lr}
],weight_decay=0.)


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
model.push_to_hub(f"{account_id}/roberta-base-mrpc-peft-fourierft")

# %% [markdown]
# ## Load adapters from the Hub
#
# You can also directly load adapters from the Hub using the commands below:

# %%
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer

# %%
peft_model_id = f"{account_id}/roberta-base-mrpc-peft-fourierft"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# %%
# Load the FourierFT model
inference_model = PeftModel.from_pretrained(inference_model, peft_model_id, config=config)

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
