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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from peft import PeftConfig, PeftModel
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import random

peft_model_id = "smangrul/tinyllama_lora_norobots"
device = "cuda"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")

# %%
# %%time
# [0.8, 0.1, 0.1] linear #[1.0, 0.2] 0.7 density dare_linear #[1.5, 0.3] 0.5 density ties #[0.8, 0.5] cat
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
combination_type = "ties"
if adapter_name in model.peft_config:
    model.delete_adapter(adapter_name)
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type=combination_type, density=density)

# %%
model.eval()
model.set_adapter("merge")

# %%
messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")  # , add_special_tokens=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0]))

# %%
messages = [
    {"role": "system", "content": "Create a text ad given the following product and description."},
    {
        "role": "user",
        "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5™ console unleashes new gaming possibilities that you never anticipated.",
    },
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")  # , add_special_tokens=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0]))

# %%
text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""

inputs = tokenizer(text, return_tensors="pt")  # , add_special_tokens=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(
    **inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1]
)
print(tokenizer.decode(outputs[0]))

# %%
