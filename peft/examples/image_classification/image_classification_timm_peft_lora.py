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
# # Using PEFT with timm

# %% [markdown]
# `peft` allows us to train any model with LoRA as long as the layer type is supported. Since `Conv2D` is one of the supported layer types, it makes sense to test it on image models.
#
# In this short notebook, we will demonstrate this with an image classification task using [`timm`](https://huggingface.co/docs/timm/index).

# %% [markdown]
# ## Imports

# %% [markdown]
# Make sure that you have the latest version of `peft` installed. To ensure that, run this in your Python environment:
#     
#     python -m pip install --upgrade peft
#     
# Also, ensure that `timm` is installed:
#
#     python -m pip install --upgrade timm

# %%
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# %%
import peft
from datasets import load_dataset

# %%
torch.manual_seed(0)

# %% [markdown]
# ## Loading the pre-trained base model

# %% [markdown]
# We use a small pretrained `timm` model, `PoolFormer`. Find more info on its [model card](https://huggingface.co/timm/poolformer_m36.sail_in1k).

# %%
model_id_timm = "timm/poolformer_m36.sail_in1k"

# %% [markdown]
# We tell `timm` that we deal with 3 classes, to ensure that the classification layer has the correct size.

# %%
model = timm.create_model(model_id_timm, pretrained=True, num_classes=3)

# %% [markdown]
# These are the transformations steps necessary to process the image.

# %%
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# %% [markdown]
# ## Data

# %% [markdown]
# For this exercise, we use the "beans" dataset. More details on the dataset can be found on [its datasets page](https://huggingface.co/datasets/beans). For our purposes, what's important is that we have image inputs and the target we're trying to predict is one of three classes for each image.

# %%
ds = load_dataset("beans")

# %%
ds_train = ds["train"]
ds_valid = ds["validation"]

# %%
ds_train[0]["image"]


# %% [markdown]
# We define a small processing function which is responsible for loading and transforming the images, as well as extracting the labels.

# %%
def process(batch):
    x = torch.cat([transform(img).unsqueeze(0) for img in batch["image"]])
    y = torch.tensor(batch["labels"])
    return {"x": x, "y": y}


# %%
ds_train.set_transform(process)
ds_valid.set_transform(process)

# %%
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=32)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=32)


# %% [markdown]
# ## Training

# %% [markdown]
# This is just a function that performs the train loop, nothing fancy happening.

# %%
def train(model, optimizer, criterion, train_dataloader, valid_dataloader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            xb, yb = batch["x"], batch["y"]
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            lsm = torch.nn.functional.log_softmax(outputs, dim=-1)
            loss = criterion(lsm, yb)
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        valid_loss = 0
        correct = 0
        n_total = 0
        for batch in valid_dataloader:
            xb, yb = batch["x"], batch["y"]
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                outputs = model(xb)
            lsm = torch.nn.functional.log_softmax(outputs, dim=-1)
            loss = criterion(lsm, yb)
            valid_loss += loss.detach().float()
            correct += (outputs.argmax(-1) == yb).sum().item()
            n_total += len(yb)

        train_loss_total = (train_loss / len(train_dataloader)).item()
        valid_loss_total = (valid_loss / len(valid_dataloader)).item()
        valid_acc_total = correct / n_total
        print(f"{epoch=:<2}  {train_loss_total=:.4f}  {valid_loss_total=:.4f}  {valid_acc_total=:.4f}")


# %% [markdown]
# ### Selecting which layers to fine-tune with LoRA

# %% [markdown]
# Let's take a look at the layers of our model. We only print the first 30, since there are quite a few:

# %%
[(n, type(m)) for n, m in model.named_modules()][:30]

# %% [markdown]
# Most of these layers are not good targets for LoRA, but we see a couple that should interest us. Their names are `'stages.0.blocks.0.mlp.fc1'`, etc. With a bit of regex, we can match them easily.
#
# Also, we should inspect the name of the classification layer, since we want to train that one too!

# %%
[(n, type(m)) for n, m in model.named_modules()][-5:]

# %% [markdown]
#     config = peft.LoraConfig(
#         r=8,
#         target_modules=r".*\.mlp\.fc\d|head\.fc",
#     )

# %% [markdown]
# Okay, this gives us all the information we need to fine-tune this model. With a bit of regex, we match the convolutional layers that should be targeted for LoRA. We also want to train the classification layer `'head.fc'` (without LoRA), so we add it to the `modules_to_save`.

# %%
config = peft.LoraConfig(r=8, target_modules=r".*\.mlp\.fc\d", modules_to_save=["head.fc"])

# %% [markdown]
# Finally, let's create the `peft` model, the optimizer and criterion, and we can get started. As shown below, less than 2% of the model's total parameters are updated thanks to `peft`.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
peft_model = peft.get_peft_model(model, config).to(device)
optimizer = torch.optim.Adam(peft_model.parameters(), lr=2e-4)
criterion = torch.nn.CrossEntropyLoss()
peft_model.print_trainable_parameters()

# %%
# %time train(peft_model, optimizer, criterion, train_loader, valid_dataloader=valid_loader, epochs=10)

# %% [markdown]
# We get an accuracy of ~0.97, despite only training a tiny amount of parameters. That's a really nice result.

# %% [markdown]
# ## Sharing the model through Hugging Face Hub

# %% [markdown]
# ### Pushing the model to Hugging Face Hub

# %% [markdown]
# If we want to share the fine-tuned weights with the world, we can upload them to Hugging Face Hub like this:

# %%
user = "BenjaminB"  # put your user name here
model_name = "peft-lora-with-timm-model"
model_id = f"{user}/{model_name}"

# %%
peft_model.push_to_hub(model_id);

# %% [markdown]
# As we can see, the adapter size is only 4.3 MB. The original model was 225 MB. That's a very big saving.

# %% [markdown]
# ### Loading the model from HF Hub

# %% [markdown]
# Now, it only takes one step to load the model from HF Hub. To do this, we can use `PeftModel.from_pretrained`, passing our base model and the model ID:

# %%
base_model = timm.create_model(model_id_timm, pretrained=True, num_classes=3)
loaded = peft.PeftModel.from_pretrained(base_model, model_id)

# %%
x = ds_train[:1]["x"]
y_peft = peft_model(x.to(device))
y_loaded = loaded(x)
torch.allclose(y_peft.cpu(), y_loaded)

# %% [markdown]
# ### Clean up

# %% [markdown]
# Finally, as a clean up step, you may want to delete the repo.

# %%
from huggingface_hub import delete_repo

# %%
delete_repo(model_id)
