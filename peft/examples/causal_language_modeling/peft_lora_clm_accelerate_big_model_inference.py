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

# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
model_name_or_path = "bigscience/bloomz-7b1"
tokenizer_name_or_path = "bigscience/bloomz-7b1"
dataset_name = "twitter_complaints"
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 1e-3
num_epochs = 50
batch_size = 8

# %%
from datasets import load_dataset

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
dataset["train"][0]

# %%
# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)


# %%
def test_preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    model_inputs = tokenizer(inputs)
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    return model_inputs


processed_datasets = dataset.map(
    test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

eval_dataset = processed_datasets["train"]
test_dataset = processed_datasets["test"]

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
print(next(iter(eval_dataloader)))
print(next(iter(test_dataloader)))

# %% [markdown]
# You can load model from hub or local
#
# - Load model from Hugging Face Hub, you can change to your own model id
# ```python
# peft_model_id = "username/twitter_complaints_bigscience_bloomz-7b1_LORA_CAUSAL_LM"
# ```
# - Or load model form local
# ```python
# peft_model_id = "twitter_complaints_bigscience_bloomz-7b1_LORA_CAUSAL_LM"
# ```

# %%
from peft import PeftModel, PeftConfig

max_memory = {0: "1GIB", 1: "1GIB", 2: "2GIB", 3: "10GIB", "cpu": "30GB"}
peft_model_id = "smangrul/twitter_complaints_bigscience_bloomz-7b1_LORA_CAUSAL_LM"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", max_memory=max_memory)
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto", max_memory=max_memory)

# %%
# model

# %%
model.hf_device_map

# %%
model.eval()
i = 89
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
print(dataset["test"][i]["Tweet text"])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# %%
model.eval()
eval_preds = []
for _, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10)
    preds = outputs[:, max_length:].detach().cpu().numpy()
    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

# %%
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["train"][label_column]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=}")
print(f"{eval_preds[:10]=}")
print(f"{dataset['train'][label_column][:10]=}")

# %%
model.eval()
test_preds = []

for _, batch in enumerate(tqdm(test_dataloader)):
    batch = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10)
    preds = outputs[:, max_length:].detach().cpu().numpy()
    test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
    if len(test_preds) > 100:
        break
test_preds

# %%
