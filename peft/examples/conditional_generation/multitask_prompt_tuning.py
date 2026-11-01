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
from datasets import load_dataset
from transformers import set_seed, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, MultitaskPromptTuningConfig, TaskType, MultitaskPromptTuningInit

set_seed(42)

model_name = "google/flan-t5-base"

peft_config = MultitaskPromptTuningConfig(
    tokenizer_name_or_path=model_name,
    num_tasks=2,
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
    num_virtual_tokens=50,
    num_transformer_submodules=1,
    prompt_tuning_init_text="classify the following into either positive or negative, or entailment, neutral or contradiction:",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = get_peft_model(model, peft_config)

model = model.cuda()


def send_to_device(batch):
    for i in batch:
        batch[i] = batch[i].cuda()
    return batch


# %%
def get_sst2(split: str):
    examples = load_dataset("sst2")[split]
    result_examples = []
    for example in examples:
        result_examples.append({})

        result_examples[-1]["input"] = example["sentence"].strip() + "</s>"
        result_examples[-1]["output"] = (
            f"positive{tokenizer.eos_token}" if example["label"] == 1 else f"negative{tokenizer.eos_token}"
        )
        result_examples[-1]["task_id"] = 0

    return result_examples


def get_mnli(split: str):
    examples = load_dataset("multi_nli")[split]
    result_examples = []
    for example in examples:
        result_examples.append({})

        result_examples[-1]["input"] = example["premise"].strip() + " " + example["hypothesis"].strip() + "</s>"

        if example["label"] == 0:
            result_examples[-1]["output"] = f"entailment{tokenizer.eos_token}"
        elif example["label"] == 1:
            result_examples[-1]["output"] = f"neutral{tokenizer.eos_token}"
        else:
            result_examples[-1]["output"] = f"contradiction{tokenizer.eos_token}"

        result_examples[-1]["task_id"] = 1

    return result_examples


# %%
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch


class MyDataset(Dataset):
    def __init__(self, split: str, mode: str = "source") -> None:
        super().__init__()

        if split == "train":
            if mode == "source":
                self.examples = get_sst2(split) + get_mnli(split)
            elif mode == "target":
                self.examples = get_sst2(split)
        if split == "val":
            self.examples = get_sst2("validation")
        if split == "test":
            self.examples = get_sst2("validation")

    def __getitem__(self, index) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)


def collate_fn(batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    input = [i["input"] for i in batch]
    input = tokenizer(input, add_special_tokens=False, return_tensors="pt", padding=True)

    output = [i["output"] for i in batch]
    output = tokenizer(output, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    output[output == tokenizer.pad_token_id] = -100

    task_ids = [i["task_id"] for i in batch]
    task_ids = torch.tensor(task_ids)

    return {
        "input_ids": input.input_ids,
        "attention_mask": input.attention_mask,
        "labels": output,
        "task_ids": task_ids,
    }


train = DataLoader(MyDataset("train"), shuffle=True, batch_size=8, collate_fn=collate_fn)
val = DataLoader(MyDataset("val"), shuffle=False, batch_size=8, collate_fn=collate_fn)
test = DataLoader(MyDataset("test"), shuffle=False, batch_size=8, collate_fn=collate_fn)

# %% [markdown]
# ## source training

# %%
from torch.optim.adamw import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score

# %%
POSITIVE_TOKEN_ID = tokenizer(" positive", add_special_tokens=False)["input_ids"][0]
NEGATIVE_TOKEN_ID = tokenizer(" negative", add_special_tokens=False)["input_ids"][0]


def classify(batch):
    batch = send_to_device(batch)
    # we pass labels here since we need to generate and peft doesn't support generation yet.
    # No clue how to get around this
    scores = model(**batch).logits
    preds = []
    for i in range(scores.shape[0]):
        if scores[i, 0, POSITIVE_TOKEN_ID] > scores[i, 0, NEGATIVE_TOKEN_ID]:
            preds.append(POSITIVE_TOKEN_ID)
        else:
            preds.append(NEGATIVE_TOKEN_ID)
    return preds


@torch.inference_mode()
def evaluate(model, data):
    loss = 0
    preds = []
    golds = []

    for batch in tqdm(data):
        batch = send_to_device(batch)
        loss += model(**batch).loss
        golds.extend(batch["labels"][:, 0].tolist())
        preds.extend(classify(batch))

    return loss / len(val), f1_score(golds, preds, pos_label=POSITIVE_TOKEN_ID)


optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, 200, len(train))

n = 1000
step = 0
train_ = tqdm(train)

val_loss, f1 = evaluate(model, val)
print(
    f"""
before source training
val loss = {val_loss}
f1 = {f1}"""
)

for batch in train_:
    if step % n == 0:
        val_loss, f1 = evaluate(model, val)
        print(
            f"""
step = {step}
val loss = {val_loss}
f1 = {f1}"""
        )
        model.save_pretrained(f"checkpoints_source/{step}")

    step += 1
    batch = send_to_device(batch)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_.set_postfix(train_loss=loss)

# %% [markdown]
# ## target training

# %%
train = DataLoader(MyDataset("train", "target"), shuffle=True, batch_size=8, collate_fn=collate_fn)
val = DataLoader(MyDataset("val", "target"), shuffle=False, batch_size=8, collate_fn=collate_fn)
test = DataLoader(MyDataset("test", "target"), shuffle=False, batch_size=8, collate_fn=collate_fn)

# %% [markdown]
# #### create a fresh model

# %%
peft_config = MultitaskPromptTuningConfig(
    tokenizer_name_or_path=model_name,
    num_tasks=1,
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
    prompt_tuning_init_state_dict_path="checkpoints_source/50000/adapter_model.bin",
    num_virtual_tokens=50,
    num_transformer_submodules=1,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = get_peft_model(model, peft_config)

model = model.cuda()

# %%
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, 200, len(train))

n = 1000
step = 0
train_ = tqdm(train)

val_loss, f1 = evaluate(model, val)
print(
    f"""
before target training
val loss = {val_loss}
f1 = {f1}"""
)

for batch in train_:
    if step % n == 0:
        val_loss, f1 = evaluate(model, val)
        print(
            f"""
step = {step}
val loss = {val_loss}
f1 = {f1}"""
        )
        model.save_pretrained(f"checkpoints_target/{step}")

    step += 1
    batch = send_to_device(batch)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_.set_postfix(train_loss=loss)

# %%
# load last checkpoint for now
from peft import set_peft_model_state_dict

sd_6000 = torch.load("checkpoints_target/6000/adapter_model.bin")
set_peft_model_state_dict(model, sd_6000)

# evaluate val
val_loss, f1 = evaluate(model, val)
print(
    f"""
final
val loss = {val_loss}
f1 = {f1}"""
)

# evaluate test
test_loss, f1 = evaluate(model, test)
print(
    f"""
final
test loss = {test_loss}
f1 = {f1}"""
)
