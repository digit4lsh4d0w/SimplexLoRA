# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: peft
#     language: python
#     name: peft
# ---

# %%
import os

import torch
from transformers import (
    AutoTokenizer,
    default_data_collator,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda"
model_name_or_path = "t5-large"
tokenizer_name_or_path = "t5-large"

checkpoint_name = "financial_sentiment_analysis_prefix_tuning_v1.pt"
text_column = "sentence"
label_column = "text_label"
max_length = 8
lr = 1e0
num_epochs = 5
batch_size = 8

# %%
# creating model
peft_config = peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="What is the sentiment of this article?\n",
    inference_mode=False,
    tokenizer_name_or_path=model_name_or_path,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model

# %%
# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

dataset["train"][0]

# %%
# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"].shuffle()
eval_dataset = processed_datasets["validation"]

# %%
# training and evaluation


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    correct = 0
    total = 0
    for pred, true in zip(preds, labels):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total
    return {"accuracy": accuracy}


training_args = Seq2SeqTrainingArguments(
    "out",
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    num_train_epochs=num_epochs,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    report_to=[],
    predict_with_generate=True,
    generation_config=GenerationConfig(max_length=max_length),
)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# %%
# saving model
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

# %%
ckpt = f"{peft_model_id}/adapter_model.bin"
# !du -h $ckpt

# %%
from peft import PeftModel, PeftConfig

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

# %%
model.eval()
i = 107
inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
print(dataset["validation"][text_column][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# %%
