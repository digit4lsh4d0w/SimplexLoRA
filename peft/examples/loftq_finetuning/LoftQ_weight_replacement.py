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
# # Initializing weights with LoftQ by replacing LoRA weights in-place

# %% [markdown]
# This notebook shows how to apply [LoftQ](https://arxiv.org/abs/2310.08659) initialization on our QLoRA model.
#
# In short, the idea behind LoftQ is the following. When we use QLoRA, i.e. we quantize the base model with bitsandbytes to save memory, and then train LoRA weights on top of this base model, we expect a certain performance gap. This is partly due to the fact that quantization is onyl an approximation of the "real" weights and thus introduces a quantization error. By default, LoRA weights are initialized such that they are a no-op at the start of the training. However, we can instead initialize them so that they minimize the quantization error. This is the idea behind LoftQ.
#
# Note that this only influences the initialization of the model. Everything that follows stays the same as always.

# %% [markdown]
# ## Imports

# %%
import os
import torch

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %%
from peft import get_peft_model, LoraConfig, replace_lora_weights_loftq


# %% [markdown]
# ## Functions

# %%
def get_mae(x, y):
    return (x - y).abs().mean()


def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def error_report(x, y):
    mae = get_mae(x, y)
    mse = get_mse(x, y)
    print(
        f"Mean absolute error: {mae:>8.5f}\n"
        f"Mean squared error:  {mse:>8.5f}"
    )


# %% [markdown]
# ## Base model

# %% [markdown]
# First, let's load a base model and calculate some logits. These logits are the baseline, i.e. we try to match their values as best as possible. We only need these logits for demonstration purposes. In practice, it is not necessary to load the non-quantized weights to apply LoftQ initialization.
#
# **Note**: We have to choose a model with a `model.safetensors` file. As PyTorch checkpoints (pickle) cannot be loaded lazily, we have to use [safetensors](https://huggingface.co/docs/safetensors/index). If those don't exist for your model, save the pretrained model as a safetensors file using `safe_pretrained` and pass the model path to `replace_lora_weights_loftq`.

# %%
model_id = "bigscience/bloomz-560m"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
model = AutoModelForCausalLM.from_pretrained(model_id)

# %%
s = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""

# %%
inputs = tokenizer(s.splitlines(), return_tensors="pt", padding=True)

# %% [markdown]
# Our baseline logits:

# %%
logits_base = model(**inputs).logits

# %% [markdown]
# ## Normal LoRA model

# %% [markdown]
# Now we load the model quantized with bitsandbytes. For now, only 4bit is supported.

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# %%
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# %% [markdown]
# Next we create a LoRA model using PEFT and compute the logits of that model.

# %%
lora_config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear")

# %%
peft_model = get_peft_model(model, lora_config)

# %%
logits_lora = peft_model(**inputs).logits

# %% [markdown]
# Let's check the influence of the quantization error on our logits:

# %%
error_report(logits_base, logits_lora)

# %% [markdown]
# ## LoftQ

# %% [markdown]
# Next, let's use LoftQ initialization and see if it helps reduce the error.

# %%
replace_lora_weights_loftq(peft_model)

# %%
logits_loftq = peft_model(**inputs).logits

# %%
error_report(logits_base, logits_loftq)

# %% [markdown]
# We can see that LoftQ initialization helped a little bit, but the difference is not huge.

# %% [markdown]
# ## LoftQ with callback

# %% [markdown]
# To help with this, let's write a small callback function and pass it to `replace_lora_weights_loftq`. What this function does is that each time one weight is being replaced with LoftQ-initialized weights, we perform a test if the quantization error is actually reduced. If it it is not, we roll back the replacement. This way, we keep only those replacements that improve the results.

# %%
# Since PEFT has modified the base model, we should reload it
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# %%
peft_model = get_peft_model(model, lora_config)

# %%
current_mse = float("inf")


# %%
def my_callback(model, module_name):
    """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
    global current_mse

    logits = model(**inputs).logits
    mse = get_mse(logits_base, logits)
    if mse < current_mse:
        current_mse = mse
        print(f"MSE improved for module {module_name}")
        return True
    print(f"MSE did not improve for module {module_name}")
    return False


# %%
replace_lora_weights_loftq(peft_model, callback=my_callback)

# %%
logits_loftq_callback = peft_model(**inputs).logits

# %%
error_report(logits_base, logits_loftq_callback)

# %% [markdown]
# We can see that applying LoftQ with the help of the callback reduced the error quite significantly.

# %% [markdown]
# ## Applying LoftQ multiple times

# %% [markdown]
# It is possible to run `replace_lora_weights_loftq` multiple times on the same model when using the callback.

# %%
replace_lora_weights_loftq(peft_model, callback=my_callback)

# %%
logits_loftq_callback_twice = peft_model(**inputs).logits

# %%
error_report(logits_base, logits_loftq_callback_twice)

# %% [markdown]
# There are further gains, but they are not very big.
