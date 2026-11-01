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

# %% [markdown] id="71GTxOD71mEn"
# ## Introduction
#
# In this notebook, we will learn how to use [LoRA](https://arxiv.org/abs/2106.09685) from 🤗 PEFT to fine-tune an image classification model by ONLY using **0.77%** of the original trainable parameters of the model. 
#
# LoRA adds low-rank "update matrices" to certain blocks in the underlying model (in this case the attention blocks) and ONLY trains those matrices during fine-tuning. During inference, these update matrices are _merged_ with the original model parameters. For more details, check out the [original LoRA paper](https://arxiv.org/abs/2106.09685). 
#
# Let's get started by installing the dependencies. 
#
# __*Note that this notebook builds on top the [official image classification example notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb).*__

# %% [markdown] id="0a_bETbqv4P7"
# ## Install dependencies
#
# Here we're installing `peft` from source to ensure we have access to all the bleeding edge features of `peft`. 

# %% colab={"base_uri": "https://localhost:8080/"} id="Z0_5BYt8hobv" outputId="aafcbc39-b972-493a-8922-2141b1621926"
# !pip install transformers accelerate evaluate datasets git+https://github.com/huggingface/peft -q

# %% [markdown] id="Y8dSVHoIv7HC"
# ## Authentication
#
# We will share our fine-tuned model at the end of training. So, to do that we just authenticate using our 🤗 token. This token is available from [here](https://huggingface.co/settings/tokens). If you don't have a 🤗 account already, we highly encourage you to do so; it's free!

# %% colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["5d2f5fb454bc4c16b520e4e96381758f", "dfd2baceac524fe29c0f4a8443b60a71", "90d8e83a6af54184a82e0b81ae7054b9", "1f96ca356b6f41b59275abe93df33f43", "eef81e9bea0c4f5d85e7efa8ebe0463a", "cab6d36980c0423fb75299c09c33facc", "dd38a658218d42d7b051c66de4d4180a", "f34be236ef9c42448ecf2957160990f7", "38deee504dab482983a8b8f340472282", "b2688e34899a449e8d1f6ddb5a66bb85", "dd4edb4de5e14dfbbee418dba0bb3573", "516c6d75bc654d62b95ac235ce84c59c", "14c23f636609458ca4493854826c1a8e", "c778798c234d45b5a4ae2f250e3706f9", "d5c5396ea2f54ff0aeb9be58b59c253b", "15bd2dcdbf4b4e74b9db09bdb8822e61", "ecf73dd75420460399bfd04d8cd81f90"]} id="31Zv6rFYr37d" outputId="6476ebcf-6d71-4b7d-ee38-dc4f8e8d024e"
from huggingface_hub import notebook_login

notebook_login()

# %% [markdown] id="AX7aJaIKjbCF"
# ## Check the library versions

# %% colab={"base_uri": "https://localhost:8080/"} id="ejkn8GBzh_DB" outputId="777afbdf-e026-43d8-8efa-80bb958d0ca3"
import transformers
import accelerate
import peft

# %% colab={"base_uri": "https://localhost:8080/"} id="A833xxo3ir28" outputId="da71ef1c-b6d7-43e2-a78b-23556785ef02"
print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

# %% [markdown] id="Po1Ve9u5v_Ul"
# ## Select a model checkpoint to fine-tune

# %% id="vhvCQpP-isJr"
model_checkpoint = "google/vit-base-patch16-224-in21k"  # pre-trained model from which to fine-tune

# %% [markdown] id="UKN3rMAsjgEz"
# ## Load a dataset
#
# We're only loading the first 5000 instances from the training set of the [Food-101 dataset](https://huggingface.co/datasets/food101) to keep this example runtime short. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 379, "referenced_widgets": ["61b957d3b51643f78a921979072fe3b6", "d7136a7b3d0040d580508fc665b9fb00", "5ee5e11191fc46dd92d4c2f1a7d6d9da", "3587d42fa09b4fcdb365956a9bb07c77", "c1ed0b68884c4d4291cd67c0e685ef18", "9102cc38ee9942ac91dc66eda069ddcb", "416c65eedcea4a6ea69dae317de79bca", "128677e1b5b14e63b06b0f81c9cc4df0", "22da54e68b1d48f9b3ba55ac1ca56873", "16c7db587b8e475fa3aa9677385b092a", "23c608994006427caca7975e0d81271f", "71f7296ec9be4d9abe1af581722b40fe", "b98e53eefc1944f193169c4f7a72b799", "1d4a5a5b7d1645a8bf8133935e173082", "d29e3b9102f14f3385e47ae6e27d1ab1", "1e3e374b08964a689cfaac9c826f207b", "b377e94780fb4e1db3b9678717e04fc1", "86103f87819b440b8464f4460f50375e", "c3178221dc074657bc0e585c4cfe326d", "3c6842e0158b4dcc9b93eddfe3279d2a", "fc612aaed5644b84959a1958b0240dda", "36c8300bcbb84627a03b94f0eea86ce9", "f0b0cad40fbd461ca7bdcdbb5f442f57", "76cf84387a7c43608ad018188eef4114", "68ef0c8550ee4c00aa8b284d48572610", "58e7f5c36d8b4836a868ce89838f1896", "9b216287b8694bcc9960a356adf15504", "4d653faaedcd497d863bbf2c429ce925", "d10f2e9c25f2417f9728aa8e43acf677", "7a35d0ddc2da4dd69068214b87bcdd7f", "9aac38a8c8694c67a34b2fced0e1a706", "9d1dff20634a403fb8829469d74301aa", "b2e64f35be2d4fa3bc95c769b78e1dd1", "fdf282b234fe4a1a8ab452ac04511b7d", "59792e1ee7074f998d5d4494c09061c6", "cd5b2433cc404ac7b1bb35c6a55f6874", "7c1b6f271fff4d60be39d291c73bfb75", "074f38bd3a9d49719188e8860fb1b5d3", "3ff84efe0edc491c884898424be4ae71", "5a79a196fd7b49128a9647347f85b364", "851fb5ac25db4bb287a6dbe948278eec", "471d44c8e49e42b89302ef53ab0eb316", "4089323832d04dc2a40e238b5fa256cc", "2d867c65533e482a96db93bc5a09b8cb", "849ac914c3cc49d29d619dd4f532d74c", "4cb3d75f80434b48beb6aa4b07c86dfe", "ff39519704b64e68b69ec06aea02791e", "e0f2599ed04c424f896e503630034e84", "1674d568877048368c842c21ffaac811", "0957e36be17c43fd89462c5d5ddcec1b", "dffe636233c84dcd9d75f34baf40fa1d", "dcefc9ba538e4da2b75f9372a4c5b5bf", "77df794cb4e4491e80ee20bbd2801a89", "7e243f4a30c645b080e688fb706b4548", "db6b68a237cf4e93ae6383448b773e47", "c580d3a6e99e48fab09b3ce799711802", "4afba780d0f244548a7f28db15b41dc9", "4e3d482feec9485590d277dfc1d0b3d3", "23436ea247dd43d8829ca143a49637c5", "9609eaf0792345b2ab457cb7188ee14a", "1839e4ed1d3c4975b34c3c050052693f", "1b31bfb0ef4c404698eb2205414170af", "6350637718344d65a757d2919de8c1ab", "42a16474e41343b2a7d46e5930b41b89", "ce16ac2b3ff244e6bd7dd58daa9f4f7f", "d25f3ebb577749d89e2e6d2a72f6ca5f", "13279e67c4d847e4846e2d34e8aac589", "d7d43177c750412cb1522eb08c01d2d9", "70b04a3579a5446f94acd422c70ac50a", "43940212a87d410c82cc9cd15f38a97e", "19fd7c60287b43bbb6e0b12c25b4b375", "9013fd35e17f44bfb7a068833adaf167", "a849dcc9c7f742d49c874597d8c693c5", "039dd9a4b99e433088a0acd8ba7b519b", "17235d013b7c4cee996d0bbc1cc6c70c", "3db63ba25e7349a785244c367d53813e", "4748461200ae4af883577e2fbb8cb686", "19ec79f5a5174aa3b26861a9662951d3", "57c15c64c2374f06a1e0a36bab953ef7", "06cf9f29b929412a8092044e25861f1c", "c2032e5054ac4604832957cb6e2e69ca", "d1ff50e1b871429a85df8cf10e73ffb1", "10c4f5677d1c4af8b3370b7fb1255065", "603dd1541db345879295edc16ace2b0c", "375ac7a15cea4ce3aa484a806cc82717", "6b6459f123ef4f24a550cd9ec3c9f809", "e3047557ae7f40e2aecccf1afad36f3f", "4fc212af0c9b45ebbe334e3dd7f11b59", "fee4fba960ac41ed97984467da41f319", "ee103846621b4c0e8e1266599b99f6ee", "dcd1c1f4fc014c4aa9ebdaf3c533a061", "a29d758fb7f147c7ad1108f140caf23a", "cb52fa97c659430a8bd71dcd76245a7f", "e7144551e74b46529b00a61f580a183d", "9b1bfa11ee3746c38155c4505abfaa86", "26520bc6555d41d9951ea0219dc4b5d7", "60472b5a360f43e89e39d641dabba57b", "aa9b6ac2785c4a5abd1189edd60698eb", "cfab815edc1f42898b656c0f4a3b366b", "c5718d031b9942f4b8bf331a8543db29", "35d862a4f00c4493920da3e2eb92b043", "16b464f168d844cba5eb0c91ab4fb91c", "af5231ecf6e2489b80cdcd435b5e3451", "62a0f83cf75d4c59a0601c5ad3a817a7", "b48f685dc91540f38690f39eace724d5", "ce4b6a4b6fec4ceb907fa436ff940bd2", "28f82c8fc9cf46c7858132a77e45834b", "ce18faf7b68140a3a8247330b356e05b", "af6a4a054a5d451b9fe256bf60a09c21", "afb1f0681bce47e1ba718900d0430f34"]} id="rI0d2_liitUr" outputId="4ae986eb-6cbb-4d9f-bb99-1ffbb05ee835"
from datasets import load_dataset

dataset = load_dataset("food101", split="train[:5000]")

# %% [markdown] id="pUjwa7G8jjgW"
# ## Prepare datasets for training and evaluation

# %% [markdown] id="-Gg9xDW22yPD"
# 1. Prepare `label2id` and `id2label` dictionaries. This will come in handy when performing inference and for metadata information. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="GC3wK2aciz53" outputId="4b065fdc-6d89-46a2-88b5-b78c2d843036"
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]

# %% [markdown] id="xgHUxR_-2-h1"
# 2. We load the image processor of the model we're fine-tuning.

# %% colab={"base_uri": "https://localhost:8080/", "height": 474, "referenced_widgets": ["9d1b9ac29dcc41e08ada578916f20a3c", "ef7c7fe37c8d459da6d20f4ccbea3fb8", "93c81a011c0a435aa90a3f4f1d549510", "da87efdf06d74b0aba268320ba7882f9", "0a0e75829d6c4031bc917ac2044d9e47", "4ee1fde44dcf49eda97e1a05173e5bb1", "a0929e66406644dbb09bbdc9c58d488d", "bcbb4d8ce16b473eae2ad03f1bea2520", "2c86eb6c67f44af590937d0f1db09333", "a100435005a34d428b9ae615f49bb1a1", "8886c333aa104900a3bb4a1904756661", "bb453686ce9f4342aaae9a9fb3500d2c", "1a4ab138be9940f081514b914fdc4623", "cc59f6643acb4054ad6df56e90d3d2a8", "236638d673934823828ee57face78184", "29de968ad50543418c6865fdf003a568", "8b9f5bca0898404b91032befbd019fa3", "e4694cffcb574863a255e9022c8ddf5d", "8a0a77b9ebd74caabb8f8a764c289a5c", "d8ac6df8420a423eb048b4db04c8925c", "920293e203f14b45b61233e1bb6f1214", "a1981bfcdb6d401e9a521e18b511cf9d"]} id="3hmq4a_fi2IX" outputId="f790d034-9efa-4a1a-e9f5-f3c6bd62add5"
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
image_processor

# %% [markdown] id="EsZYbWKS3cPK"
# As one might notice, the `image_processor` has useful information on which size the training and evaluation images should be resized, stats that should be used to normalize the pixel values, etc. 

# %% [markdown] id="jKFuKh9P3E-e"
# 3. Using the image processor we prepare transformation functions for the datasets. These functions will include augmentation and pixel scaling.  

# %% id="Zj33iIoCi3Uy"
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


# %% [markdown] id="X4IPqOeK3UKW"
# 4. We split our mini dataset into training and validation. 

# %% id="_uplVC66i5Gd"
# split up training into training + validation
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

# %% [markdown] id="KV5Mlf4e3X5K"
# 5. We set the transformation functions to the datasets accordingly. 

# %% id="0QuiqyiXi9fN"
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


# %% [markdown] id="sA1Cq97Ijpp8"
# ## Load and prepare a model 
#
# In this section, we first load the model we want to fine-tune. 

# %% id="Zxgrg45Xty2S"
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


# %% [markdown] id="TYjC-A-44bHO"
# The `get_peft_model()` method that we will use in a moment wraps the original model to be fine-tuned as a `PeftModel`. So, it's important for us to initialize the original model correctly. As such, we initialize it by specifying the `label2id` and `id2label` so that `AutoModelForImageClassification` can initialize a append classification head to the underlying model, adapted for our dataset. We can confirm this from the warning below:
#
# ```
# Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.weight', 'classifier.bias']
# ```

# %% colab={"base_uri": "https://localhost:8080/", "height": 176, "referenced_widgets": ["9397ebc3ad2e4141a1405bb1bd0aa315", "6f19448725b84be4bacc3b699cd065a9", "864b6bb42f0b46a2a7bcd0d8cbac3837", "5634fd283a9e45d9a55c02ca1b7c784c", "87aad727ec964c9d97346ac02ed0caae", "3c33964c8d804600ab5a26d0717c508d", "f5041033ddf94f459ed8d1747f6b2d6e", "c1af5e6c4259480eac652f6c6269ff5f", "194dd0bcc350480c9ddd3e4ef17efc3a", "4a44332ff1224a19a5f1c18e2b827759", "7e1ac6f28fb340d3bde1e7b4893bb0aa"]} id="3J5DokIqi-wV" outputId="66275479-0491-4db1-c265-333609b2dde2"
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
print_trainable_parameters(model)

# %% [markdown] id="1EqYCiTy5F9N"
# Also, take note of the number of total trainable parameters of `model`: it's 100%! We'll compare this number to that of the LoRA model.
#
# We now use the `PeftModel` to wrap `model` so that the "update" matrices are added to the respective places. 

# %% colab={"base_uri": "https://localhost:8080/"} id="LNASJrqoi_8-" outputId="8088d2a6-b6fb-4ecc-f7c3-8f0797f4f6ff"
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# %% [markdown] id="OKQeuUDhjC3E"
# Let's unpack what's going on here. 
#
# In order for LoRA to take effect, we need to specify the target modules to `LoraConfig` so that `get_peft_model()` knows which modules inside our model needs to be amended with LoRA matrices. In this case, we're only interested in targetting the query and value matrices of the attention blocks of the base model. Since the parameters corresponding to these matrices are "named" with `query` and `value` respectively, we specify them accordingly in the `target_modules` argument of `LoraConfig`. 
#
# We also specify `modules_to_save`. After we wrap our base model `model` with `get_peft_model()` along with the `config`, we get a new model where only the LoRA parameters are trainable (so-called "update matrices") while the pre-trained parameters are kept frozen. These include the parameters of the randomly initialized classifier parameters too. This is NOT we want when fine-tuning the base model on our custom dataset. To ensure that the classifier parameters are also trained, we specify `modules_to_save`. This also ensures that these modules are serialized alongside the LoRA trainable parameters when using utilities like `save_pretrained()` and `push_to_hub()`.  
#
# Regarding the other parameters:
#
# * `r`: The dimension used by the LoRA update matrices.
# * `alpha`: Scaling factor.
# * `bias`: Specifying if the `bias` parameters should be trained. `None` denotes none of the `bias` parameters will be trained. 
#
# `r` and `alpha` together control the total number of final trainable parameters when using LoRA giving us the flexbility to balance a trade-off between end performance and compute efficiency.
#

# %% [markdown] id="mRbdQvEujHHP"
# We can also how many parameters we're actually training. Since we're interested in performing **parameter-efficient fine-tuning**, we should expect to notice a less number of trainable parameters from the `lora_model` in comparison to the original `model` which is indeed the case here. 

# %% [markdown] id="m6lBFL_D-w7k"
# ## Training arguments
#
# We will leverage [🤗 Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) for fine-tuning. It accepts several arguments which we wrap using [`TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). 

# %% id="-iD2F33JjIzC"
from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
)

# %% [markdown] id="7_SA1HcVAUNP"
# Some things to note here:
#
# * We're using a larger batch size since there is only a handful of parameters to train. 
# * Larger learning rate than the normal (1e-5 for example). 
#
# All of these things are a byproduct of the fact that we're training only a small number of parameters. This can potentially also reduce the need to conduct expensive hyperparameter tuning experiments. 

# %% [markdown] id="XOlDXQnrjuc_"
# ## Prepare evaluation metric

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["86fca0e29e7a4dc8b2234134014958f8", "8378c214cd044bfb97c452d811df748f", "828a652d92724ba4888d924846a79374", "cfd59ddfe85f4585865df8df47fd491f", "94f39a2f3baa4bb2bffd1b99e8a31367", "b6a1b7db4afe44c792907f6377cde35c", "dca7d0a0d2aa479083d81a54489d3717", "4bb8b2d7000f464ba3ff18ce03fcfef4", "793ebaa3acc6482bb135ca0ca864be4d", "bcef9cf2b00c46878f07c48875f7d194", "47659b15eb284f06bf9735ca2e425646"]} id="guYecwzyjLmj" outputId="7efb445d-a442-4173-c869-9bc5be044e2b"
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


# %% [markdown] id="mNeLDXaE_989"
# ## Collation function
#
# This is used by `Trainer` to gather a batch of training and evaluation examples and prepare them in a format that is acceptable by the underlying model. 

# %% id="qIicZRMrjNC3"
import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# %% [markdown] id="YpWudVaJjwkx"
# ## Train and evaluate

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["95449e7030324f99b148bbaedc15c155", "be1ae63f3e804e23abe7739e9f4577fb", "d5b95aa9cab446f88d61e9f4a25a8e2f", "2e31d27cc694434aa869896041c72bee", "4a3ff00e64b548ce89355778907e48c9", "d947ec84b16c4781959427b610328ab9", "9123141f7c164d458a21e54fc579fa66", "83d6fbf463264c71a4ec8775e26c7c38", "f02443fbda394fefa162f4ff5b2d2ce7", "0d75dd458e3448a58ea4e19c28e787c0", "0bc5c81047994f5b976a927b8ed47cbc", "da13543779034424aaf6f5c4a96f0457", "2d13b401dcf94089a4a78a62f05bdce3", "dbc00727fa1c4e00aacf627c04527649", "e0d98c36e5d242b2905adf8167ac348a", "483b46ed1e8148498d54e4d6f4c0ca8d", "070fdaf418de43a3a5ee0592e8aca103", "62818f9421694139bbe1d9ad6e822b10", "3efccb526dec44bf9801ac13dcc1068d", "8ecde04d15ab47f9b78d561615ca567d", "ddf88cbfaaef4a55babf480816db7d28", "38f30da546444f8199673003d0a92dda", "e4a4122ff32a41a1917459709224fc6a", "8ba66e043f8a4975bd77ecd343401260", "2e0bb2dcd85640d7b85d80469ea9f9f3", "1c208beced884b9291c5bcb7b4f71680", "3f188d6d34774154afc297b13a3eb9e8", "1f8c65025b63466192897a32a92182e9", "ffc13c11355b46bb9cafcb17f3e1535e", "136a56e1f70c431bae0a3ac01751a814", "7a3daf19ee744c7b8baeb028db05009a", "63c07a01593f467f9c0e7c5e283d58ae", "972f831792cd4e89af109462dd5b9210", "098770d5a36540dea54d27d7fa9bcd56", "5da11a8c37ae41458ea4491ccdfb4db8", "23ec20ee5f0e48be9470415810cd0b4b", "e88c3ad56ef24e4d8281898b08ff6f4b", "c5670295387a4c199571a2a21a6b69dc", "6dac696d99a44ea399a1bd5e18f08428", "b3a8eebed60f4ecab7d508c976e2e56b", "d7c394bc6a3249e9b3fcbae2ebd25eb7", "fea27a80cd2f4b4dba84ecdfefd2722c", "1fdc59cbb8724c618ce6e586e2c9723f", "dbaa70ad4f1d4496a670601fe447116d", "fd9df81594724b88b54b4e3e1b19370a", "c8267c689fb14afc9a8eb3ecb6f4fd4c", "adb09cebab13484a8d75a338eaba7b0c", "d68194cf7d264df7820f27eb4d070de2", "8bf8a843d65142bbad81de74aa8573f6", "fc9d0c314ca14826993fe1f24b070b5d", "bc71a433928e4870b56a3d81e35e6351", "9fb3579ca9714141a7857a513c379f03", "edc0742a08a445a594139200c7f03c60", "d9329cb3c1704691b6a36c293bcbf41b", "dfa468dd89174d97bcaabbda0ed8e117", "c2f4b407a47f4d958986035188c6ece8", "75b76841f06249a0a77c7e38b85a14c8", "79351db5d2e1468e9b91d7bd2274612e", "c6ce0e9bdd90400f9cf2debf9165758c", "138198ec50a9494889319d6c94da92bd", "e6ce3e626b1744c7ba3da26d1fde5fa5", "8e1237963bb5479f93318c5cdc6a8593", "2d8a53b2a2bf42b9aa9cb2d9978ccee2", "69dcd4770fbc428cb56498b6577e237e", "cb3ab56aa43e4b94b978764caa6057a7", "2e24b7250ee04fbb810e5d6ade107c51", "2ab85fcc8de042d0bdb9ca79b8e404a4", "77deda3ef342432f9b0f684a9b32e248", "f3991aaad13a4c50a7809483b7907b7b", "eca3b1f4ad76430483a221470e592c13", "a1d5bc95f1e24e3293414e08aa5c8bd5", "b48fcbe51098482aad8798670111d60d", "7654b707840e4afb9bab8218418fd096", "4a118fa87e424664a2d2ed7c7f58f3fd", "852b01d8592b4d8aa2c4297d6cf75f78", "e302923a9df24e5fa8fff79c203ead9f", "f7b9abca32ec42edad5ec6e52882f732", "923cf8641a7946f69ff41fb88b2b86f8", "9d14ba8675fb4c689dd821ce7794abb6", "6932d2462135413cbc293964eb1c8317", "7d5831ee2a1c4e649f5508631d64e7cc", "70af909af7de4161b4a72a8e15d116f3", "d04c1c4d04fc4928b4a2a0e860f996e0", "98d32ec7fbf54effadb886bc4ec6ce79", "3c0cacee5997480cbedc0e9d59a62544", "fee3db0deefb410db4c572efd95575bf", "2eae62f1cc46449dba93f5eda0cb3f1c", "e8026bcb0e2c4b14bc6c84537c8c4ae9", "7bd32cf88c154303a76759d674795856", "a907de6474cf45cb91b3f2efc40821b9", "5dc4129160514a479ffb2f0564aee071", "16620105b32f434eb77b0df56ed49e45", "7b8f0cbb552447549aed602f937fcfb4", "3f6394cb0ea242f28c4ba6b3b2d37e9f", "02a0b01d31a34a1c924786037fecba09", "e4074e524a19455fab810ec454fe8bf1", "d67dc70cfc9246f79a59261a69b28b41", "edb0d1ba5e114af9b6705969f58ece7b", "1954a636239b40169659e2ae8ef3b127", "d9c15769da2b49e4b67d43d95be30cd5", "4d1f6114d4034f758bf8cc35485e0056", "223a13f77e2e49a09660890eb4213b30", "1639075b181f4945ac32af116b22d1d7", "ad6adbe84ac940ffbf89017a269a3e75", "0ba38362cf8647c08b0beb21a2c39442", "65a0aed816c84164a6ee6a41d300fad0", "47cf3db935ba4e109843b03a9577c184", "be1ec4b9b8964810867b0e00bcc4868f", "cf024daa51f74777b98028df10dbc9c5", "eaf2c76a172d4da6846c6face18a3b58", "bda7e5662f2e4fa292752efd4947c5f6", "4825c09098e1446a9ed3b653b77894f4", "f544720498e44c49add78550b46edb3a", "24b3737dc76c4d4f9ba2603c653a3ce2", "2ab99fb38f8d4bef85d9833bb628fa00", "148c9912cec5473bb6f8533add143cd3", "92dfb889fd22439bb7b5fd31e4991c93", "9281c5aec5b84411a05e4762125388d9", "f5b5d6ace35a4a82bfcf2549b93c8558", "397dc640630841d7845bf5a8739ce5eb", "07f5e653fe6740e8a71fb9de101884f3", "ea2217bba8574c7890a411f27da0c147", "b1b6922df40c4af69b00b4e85db770c4", "2b8bc04ac3104592bf950e349c034c2d", "cf815c0979644cd6ad2c681fa96c0648", "7541b2304cc5466cb2369c0025d2d243", "f6a9243d46cb4c0fbdf3f80f7074f6c5", "a2b51be9304342e39431b82957eb4b25", "5e51957908eb48489357a7c3924ec5c7", "41320a22032c4884affc456f7c6db1c1", "6940a405215c4e2caadbe209c677bde0", "b21331417d084aba80f919b71933bc2c", "c910ae80ec1a4718915e9a861215f27c", "245c5418ca084fb6bc0b027576a1f789", "d0bc0e6038eb46dbbc5f5593d4c285ca", "a450c318d99a477c9f7341458ad4bc8d", "02ac19466e24404a92e769ed60604881", "431174d906f640baa17842fdb3a8714b", "638b918aaacc4c4782b9e16ca66549e8", "7c038ffcc1dc4e3fbfed17d94327353a", "91f6edc592394a0bad250e68d3c22017", "4533e8ce655649cba93553c8a2b17f37", "63b53da916fe479e8cd495eff8d16df8", "be29ee88a7ec489b8320f7306d78931d", "a1df731c5c5f4f9cafa19323a750ebea", "bf2e140f54d74df09663d3fcf1660d0c", "acae77f181ed43a1b29412c575435a7f", "87581c98cd174bb684ec259066d047ea", "260fdda06c214ed499f69fac4077d476", "aed3b4e6110442398c25d37456b78b5d", "8da58936a6e64529af9a3e3f314e49cb", "9a5e108d8b5a41ae95a619bfc6c8f3a9", "e54b7fc2f9b94118ab97f2736862f77d", "dfe97442852c4338843c65333b25623d", "5b7be0df4db54866a3b6ef9204ba5a89", "cde9d5cbadf14a5abe294dba0fa5bd2d", "835db77232e74cc18a6b5db2ace40bfd", "6099227eddde44009582b9f24fc96150", "9eb912a195f3461297b7143cb1b04678", "6f1a325b02f54352a0b412d7f4420bbb", "e8c2cfdaf0eb413189d93924eae757c7", "d2469e1f1daf4d4cb0faf35ce90f6445", "e667b14a3c0e41c6a16c4be453f10378", "af7ee2bb7ccc4c00838a2c6b937e4e8b", "84c281446c5b424090a5eecbd733b050", "fdb3673fdbf24468a9965f13196b78ed", "0b82dbc29d514f4e9e012fd755948e52", "af1a42626ba7452189fbb5987b159b9c", "93788683ef8e4c71bc1c0b3b9cc7219c", "6e4983016e4f465b85ab7a472d0e986e", "aa68207e72b0467cb9a4354dc231db2f", "60e6952873524186aad05661a00bd240", "d4aa1670fdab463bb0a0e6fe104988bc", "8934f66530644f0882e292bfd5458b0f", "a2671f512e404f64bfa3f376449f6947", "a61a30ebaac846c1b7a03c6a93127aad", "bd9e7cb0f25445739ebcdff0d3112052", "52d00532eeee40aa91e8a5c2a10e50a7", "21c75049df804ac4ac7bc6349a639056", "041f73c9a038411aa6d59cf8a93f6d47", "b8e180259fd94096884f7e48a53b0fce", "7a65e650113e476a8cb66caa92973dd3", "d0c95a20c2664c149886b72fa665d3cf", "5fd1cd8bf125446a96b9438fbbe52710", "eb864284052c46b28b93fc79bfed740f", "de92f68231294aefb249f400475bc9a4", "f981fb4aae504045aa10889dceeb6cac", "77f7230186b14c628f5094f9fd8d82da", "855a0f70b9ac489a86b53792e119329a", "9005e9db560d4e89880bdd18403ef9e1", "070a734e268045098977db14c6565777", "91ec8a3f10804d629cdfd47c61411c91", "0465571b25714ecda9dfe6ff1a495a87", "dc078f0db3e54199bef0c11ee5e6297e", "c3f7788abe754cb3bfbee3fadda54916", "fe93399cc15f4f29b6a37f6a65cf8c9b", "ad2068dd9c2040f6ae44bc873fa7b6e7", "9434b43de9954c83a4311432bdd68376"]} id="p2-RStfgjOQt" outputId="6a1d0384-40ce-40b2-cca9-91a915d5b939"
trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()

# %% [markdown] id="b2NENHxHCejv"
# In just a few minutes, we have a fine-tuned model with 96% validation accuracy. Also, note that we used a very small subset of the training dataset which is definitely impacting the results. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 198} id="_MAd2906jQKG" outputId="7b825531-4e8d-4666-d6fc-eaa1accf3bb6"
trainer.evaluate(val_ds)

# %% [markdown] id="qo_scDEyAQER"
# ## Sharing your model and inference 
#
# Once the fine-tuning is done, we can share the LoRA parameters with the community like so: 

# %% colab={"base_uri": "https://localhost:8080/", "height": 172, "referenced_widgets": ["980d2f61332f414d9888b76e78774be4", "cbc0ba8e49a740fcae7b94fe7edb8107", "b3f07ef160a7425880ffe362008d4400", "49ed3330fa9645eda8b5aed0fd7cbafe", "f96302d0c2d849c5b5a0206b65e461ab", "fcc7ad16a0b14d96acd9be9e03ac6af9", "1a08961f063346ccae206a863ab7df6b", "1150e391e753424da7d65bda10463da4", "e6e36d744e1244aeb7eb0c4ce392372d", "b2ad992db5a045668fa55c7393ec7870", "b5fad0f3f2d543ecaed726e52d2d86bb", "e83fd078f467406da0baf26e18b39e89", "9b4b67731a7a4bc59be132b53c24eae8", "e33243b001274d02a25f5940ba41ecf6", "06e4c619e366427a8ff4c358196ecd12", "bacd429b42d843299cb75224db3afb1e", "c53429e699e64b3d8895a355bbd947a6", "2bfd04824f2e4fd6844dd38e46dbbbdf", "40a5a50aeca24f0d8990da97971004d1", "be71e6438e0e49759d2f72feec520cae", "1902fbc4c1da4ffe90f0947e58eb5d48", "12a94351242444d7b0d23b8accc1824a"]} id="TyQvIcnFzLIV" outputId="7ac2819e-080e-4940-9755-15c32832d9a6"
repo_name = f"sayakpaul/{model_name}-finetuned-lora-food101"
lora_model.push_to_hub(repo_name)

# %% [markdown] id="uCmWERkhSRNO"
# When we call `push_to_hub()` on the `lora_model`, only the LoRA parameters along with any modules specified in `modules_to_save` are saved. If we take a look at the [trained LoRA parameters](https://huggingface.co/sayakpaul/vit-base-patch16-224-in21k-finetuned-lora-food101/blob/main/adapter_model.bin), we see that it's only **2.6 MB**! This greatly helps with portability especially when we're using a very large model to fine-tune (such as [BLOOM](https://huggingface.co/bigscience/bloom)). 

# %% [markdown] id="Kt57qe42Sr6E"
# Next, we see how to load the LoRA updated parameters along with our base model for inference. When we wrap a base model with `PeftModel` that modifications are DONE in place. So to mitigate any concerns that might stem from in place modifications, we newly initialize our base model just like we did earlier and construct our inference model. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["323eb0d9dade4c4fa3a9ad2b973dcbe1", "484f7a55438c47f59365242c4753edba", "c5665d0bc652405c8754474871baab06", "82f96ac9299a4841a550ad3daa0099d0", "f031aaf7fbc648a7b8a2e5faf37df14d", "f22598cf4ade4427a1b437fd45aabcc4", "0cd35b1092064f42908ce4123b79a8af", "8b7713310a814991aec7929fa715ec7c", "fcc8254622324f8ea965e12e4d4966cd", "3a71257db7bc408d8e4d1fbcaf1dff93", "ccd34ccf2c864c609a0b4fcee7327b31"]} id="IY78Ty570etw" outputId="8ffc4da6-8165-46bb-dd4a-a79b9f236cb3"
from peft import PeftConfig, PeftModel


config = PeftConfig.from_pretrained(repo_name)
model = model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
# Load the Lora model
inference_model = PeftModel.from_pretrained(model, repo_name)

# %% [markdown] id="F9QDze9AXH8n"
# Don't worry about the warnings, they're harmless. 

# %% [markdown] id="Qb19xGKDW1XS"
# Let's now fetch a sample for inference.

# %% colab={"base_uri": "https://localhost:8080/", "height": 200} id="HxJAOWPGU1nL" outputId="ab0e4295-325e-4f21-b0c2-c2b09dd05a7a"
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image

# %% [markdown] id="dFuqZgmCW4cu"
# We first instantiate an `image_processor` from the underlying model repo. 

# %% colab={"base_uri": "https://localhost:8080/"} id="dN4x_pj-VQx8" outputId="b4fddd49-2f10-48e2-de31-0cefd20d405d"
image_processor = AutoImageProcessor.from_pretrained(repo_name)

# %% [markdown] id="Dc0rCwC5XAaL"
# We then prepare the sample for inference.

# %% colab={"base_uri": "https://localhost:8080/"} id="57C6tcdnVYu1" outputId="f164dd91-8679-482e-fb10-2f530d4ea9f4"
# prepare image for the model
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

# %% [markdown] id="Pn4T1GyTXC47"
# And run inference!

# %% colab={"base_uri": "https://localhost:8080/"} id="YyQznW5WViMc" outputId="d1b7a77c-68b3-4f6b-a945-4b32e85baabe"
import torch

# forward pass
with torch.no_grad():
    outputs = inference_model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", inference_model.config.id2label[predicted_class_idx])
