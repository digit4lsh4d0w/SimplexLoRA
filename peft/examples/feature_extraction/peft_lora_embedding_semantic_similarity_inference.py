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
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from tqdm import tqdm

import datasets
from datasets import load_dataset, DatasetDict

import evaluate
import torch
from torch import nn
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel, default_data_collator, SchedulerType, get_scheduler
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from huggingface_hub import Repository, create_repo

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from peft import PeftModel

import hnswlib


# %%
class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)  # , quantizaton_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"":0})
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs["attention_mask"])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


def get_cosing_embeddings(query_embs, product_embs):
    return torch.sum(query_embs * product_embs, axis=1)


# %%
model_name_or_path = "intfloat/e5-large-v2"
peft_model_id = "smangrul/peft_lora_e5_semantic_search"
dataset_name = "smangrul/amazon_esci"
max_length = 70
batch_size = 256

# %%
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
dataset = load_dataset(dataset_name)
train_product_dataset = dataset["train"].to_pandas()[["product_title"]]
val_product_dataset = dataset["validation"].to_pandas()[["product_title"]]
product_dataset_for_indexing = pd.concat([train_product_dataset, val_product_dataset])
product_dataset_for_indexing = product_dataset_for_indexing.drop_duplicates()
product_dataset_for_indexing.reset_index(drop=True, inplace=True)
product_dataset_for_indexing.reset_index(inplace=True)

# %%
product_dataset_for_indexing

# %%
pd.set_option("max_colwidth", 300)
product_dataset_for_indexing.sample(10)

# %%
from datasets import Dataset

dataset = Dataset.from_pandas(product_dataset_for_indexing)


def preprocess_function(examples):
    products = examples["product_title"]
    result = tokenizer(products, padding="max_length", max_length=70, truncation=True)
    return result


processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Running tokenizer on dataset",
)
processed_dataset

# %%
# base model
model = AutoModelForSentenceEmbedding(model_name_or_path, tokenizer)

# peft config and wrapping
model = PeftModel.from_pretrained(model, peft_model_id)

print(model)

# %%
dataloader = DataLoader(
    processed_dataset,
    shuffle=False,
    collate_fn=default_data_collator,
    batch_size=batch_size,
    pin_memory=True,
)

# %%
next(iter(dataloader))

# %%
ids_to_products_dict = {i: p for i, p in zip(dataset["index"], dataset["product_title"])}
ids_to_products_dict

# %%
device = "cuda"
model.to(device)
model.eval()
model = model.merge_and_unload()

# %%
import numpy as np

num_products = len(dataset)
d = 1024

product_embeddings_array = np.zeros((num_products, d))
for step, batch in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            product_embs = model(**{k: v.to(device) for k, v in batch.items()}).detach().float().cpu()
    start_index = step * batch_size
    end_index = start_index + batch_size if (start_index + batch_size) < num_products else num_products
    product_embeddings_array[start_index:end_index] = product_embs
    del product_embs, batch


# %%
def construct_search_index(dim, num_elements, data):
    # Declaring index
    search_index = hnswlib.Index(space="ip", dim=dim)  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    search_index.init_index(max_elements=num_elements, ef_construction=200, M=100)

    # Element insertion (can be called several times):
    ids = np.arange(num_elements)
    search_index.add_items(data, ids)

    return search_index


# %%
product_search_index = construct_search_index(d, num_products, product_embeddings_array)


# %%
def get_query_embeddings(query, model, tokenizer, device):
    inputs = tokenizer(query, padding="max_length", max_length=70, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        query_embs = model(**{k: v.to(device) for k, v in inputs.items()}).detach().cpu()
    return query_embs[0]


def get_nearest_neighbours(k, search_index, query_embeddings, ids_to_products_dict, threshold=0.7):
    # Controlling the recall by setting ef:
    search_index.set_ef(100)  # ef should always be > k

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = search_index.knn_query(query_embeddings, k=k)

    return [
        (ids_to_products_dict[label], (1 - distance))
        for label, distance in zip(labels[0], distances[0])
        if (1 - distance) >= threshold
    ]


# %%
query = "NLP and ML books"
k = 10
query_embeddings = get_query_embeddings(query, model, tokenizer, device)
search_results = get_nearest_neighbours(k, product_search_index, query_embeddings, ids_to_products_dict, threshold=0.7)

print(f"{query=}")
for product, cosine_sim_score in search_results:
    print(f"cosine_sim_score={round(cosine_sim_score,2)} {product=}")

# %%
