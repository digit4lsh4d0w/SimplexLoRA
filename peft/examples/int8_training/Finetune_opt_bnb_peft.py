# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: 'Python 3.10.11 (''accelerate'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown] id="WE5GJ6s7y0Xo"
# ## Fine-tune large models using 🤗 `peft` adapters, `transformers` & `bitsandbytes`
#
# In this tutorial we will cover how we can fine-tune large language models using the very recent `peft` library and `bitsandbytes` for loading large models in 8-bit.
# The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA), instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model. 
# After fine-tuning the model you can also share your adapters on the 🤗 Hub and load them very easily. Let's get started!

# %% [markdown] id="TfBzP8gWzkpv"
# ### Install requirements
#
# First, run the cells below to install the requirements:

# %% colab={"base_uri": "https://localhost:8080/"} id="otj46qRbtpnd" outputId="2aa109f6-3f4e-4887-a16e-336f51e7cc9a"
# !pip install -q bitsandbytes datasets accelerate
# !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

# %% [markdown] id="FOtwYRI3zzXI"
# ### Model loading
#
# Here let's load the `opt-6.7b` model, its weights in half-precision (float16) are about 13GB on the Hub! If we load them in 8-bit we would require around 7GB of memory instead.

# %% colab={"base_uri": "https://localhost:8080/", "height": 408, "referenced_widgets": ["d4de260ffd8a440eb87eb900fc1bb1d3", "8602b545a9f8474dbb3cc178ac0b8e60", "b46919912ee54f6f9f2ce9080be1c61a", "50374e3ab81c4626a182e61fc03b94ce", "2144bc2897dc40b29f060e30ace12275", "949ca70002ca4472bbc21fea4d7ac745", "49943c9dadca43a584b3f354ba45280c", "6123e53fb26b41f0af9a3a3348ae1afd", "285ef943d540400ab827c462945a259c", "95727290446244ccb9626f4594949675", "61b54aa6c9e94ee1bc45c15a9e3f7917", "fc2d5ffe254d425b939252ec46ec27cc", "f65af2e868244edeb0cc9402534874a8", "e466054f08004bbcabb24e400cb3c7fc", "6ea40800dfd849e3b106bae71fc53ae3", "722a01f42b7d4c38836a4546ecb38108", "1bd5179cdb474b65aa06eca3520ad37b", "04d367124a3b419ab1fa1dfd4f9004c3", "4ea667f48b9f4e1f9da7c5a0d3025b85", "9b96c63630654773acd38b8b88371f28", "cb69ae47666a4603a07a8778e2ae7d6e", "8f5e9f2d11d54fd2a08dcbff9f6da05c", "c6f712eadc4d49019b2bd355968cc155", "5fd979c05fd34311af877ce1a988ead8", "179165ff4c0e4586aaf3a40b8502f428", "cad1d8326e474a4f9ee13db9005121dd", "b049ac0b3d2e44ba9b1fc1cb2dd3de62", "921c268ecd724b6a8869dae8f81d558a", "4cf85a85c0764f15b5134c81b360e910", "5936d9eb59e147eea6482006decfe0ee", "fad61c68edc84c8197afeafded84280d", "451309d8b3ca4cd9b6f538640477039e", "c7621e14aa16421d9758321e433b92e4", "5aa74b9b30614172b07f88873cf89471", "cf39fb025e3d4635a5695135a56d9f64", "deaf9732f33446a3be015d2ec16aba76", "bdfd856dd8ee4ae09205ad9d1b9cc806", "7a49f7a55b054b6d829f290a1a426a7a", "5b695486a0ac442b8b6a8ef2ebf4e57a", "03be7cb91cba4afab795aab7aa242ee1", "8f7890f54d514f80b1eb17905cf0f964", "ae660d51267544a89f0ad199cc12b6fa", "d5bb8d7359274c8e9cc79df563175137", "98d5a80ec50c46c18f9ee991e2982115", "e73e5388182040a8937ccf1748171a87", "f797f7b8b13f4b3cb871522d34498631", "41f47c864e094cacb1c550d37ddfe80e", "104d982b444947fe8e4fbb2c2f082616", "2cf8581583c641fb95d1e16aff7d4cd1", "11808e9097424dd0b22a5af6c77813f3", "dc08d237860e4788a8ceeff4518c2612", "7bbf2c40a4ae46fe9e5885db08975263", "f00cbc4a89f5492787ec489da65ff70b", "8a7cd2194113493a841b00d034b5f1ba", "e9673363e85448d494ac9ac5d7ba0efb", "a994beafbf3f4c20880a7bbe3898db36", "7769c261781f4f5483c7e9d58c1a5573", "4a713a8fe16f4e81aa841c69711fa136", "08d036904ecf47ed88e129ba6e2b285c", "9f1182fdddab43b59ee98bc701965a17", "c2302f535b114ad780bfa440445c2e28", "8fe032f285ba4858b8efd68119c217b0", "4ca695fca3d140c6ab4e1e1d34df807b", "b77ceaf55dc04810963cdd01126478f6", "0ccdefcb25e14d229b2634ffae4a6d3d", "11d6b952503c4824b36b66e228f87599", "1e9391f6c89c4d08859ef3413edb19be", "039bbda2402f469eb21ba7ec7ea589a6", "5da6eef8fb0048219159f38a68727b64", "c006e62ee6d04831b2b89273ef04a8ac", "af0634c2539a43989902daef47776901", "62fea00ef0364af287e6097b964d00c4", "9e5ef73f8b244845a6b9002fa5c35d15", "5cb09c84e1e144e0a92580fb5e1ce2fe", "0f561c1660744251a8f710b69d434c87", "9e5f870ef80242f5af09fad70f84ea62", "6f173ca73dd545deb22b8cd0470d925f", "4e6d5943bc374b388b93ed115e44b6a5", "cbe2a6ea41834e95a27d6f02c3c0eeec", "7beb5f4efefc4593abac253df74d1405", "370cefbaeebf41f582b7507ad493055f", "e3df9dda16e244aa9b61d54f9e21ef40", "23d2ad64a17041b7a006dad1e041e0a1", "c1fd6a1234274a44b838a09f3f5380c6", "81bb51d088374394becd9a45ec3b17d4", "9e762779e5434bb7afcc295b61c2f4e4", "f539b7a4665449de9eac209a20629969", "32d528db79ad4f6f836ab2e0df5ac426", "1ca7684b79c5438fa06b047bd2b3283f", "a07688185bff4c4b8cbed3af3b4cf802", "0272f1d9f93f4dd788363a8409cdfd69", "27b41d23d2c64127ba3ae8464958f855", "2b54032c0d8e4a2897aed1ac1c79af14", "ed8fa1048e814f2fa3666899fc42e55a", "97daf559100c44ac983562fea93c5fac", "72e511b775604d899ff5b3fa2ebe9fc4", "da946f86590447d2ab98b9da468fa66b", "54fe79d5c7254117a2209927a7248dd4", "1d122e4eaad54e06961288484f31e18b", "d46b5725c35142a89617e46c0e8d3679", "c5493c23fd5542738ffd1ff5f09a6a67", "a1a80d3460984c2496ada5a634875934", "598c5584ffba4f26815c4e87bb1595c4", "fe93f25323604447be0bb1d24a0c2c59", "00c2e2d3ee8b45818ba84da12c6b11e2", "6ccea64c2e614a9fbdcc2f716cecaea0", "63fc9a9eebca4f2db2ed8a385fc5e204", "9a4860dfeac944db85e6e532599bc1cb", "3b946e1bbab24629b98307275fbe7cbb", "d9d36f8ff5f747bf90fbc8a7d35a6664"]} id="cg3fiQOvmI3Q" outputId="135a7675-6a4d-4786-b5dc-34cb867f40c7"
import os

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", quantization_config=BitsAndBytesConfig(load_in_8bit=True))

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# %% [markdown] id="9fTSZntA1iUG"
# ### Prepare model for training
#
# Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_kbit_training` that will: 
# - Casts all the non `int8` modules to full precision (`fp32`) for stability
# - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
# - Enable gradient checkpointing for more memory-efficient training

# %% id="T-gy-LxM0yAi"
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)


# %% [markdown] id="KwOTr7B3NlM3"
# ### Apply LoRA
#
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

# %% id="4W1j6lxaNnxC"
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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# %% colab={"base_uri": "https://localhost:8080/"} id="4iwHGzKBN6wk" outputId="039f7175-14c9-42b4-a078-80d27aab161c"
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# %% [markdown] id="QdjWif4CVXR6"
# ### Training

# %% [markdown] id="b_RPiO7f3ClX"
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["f357166c6e5f43f39d0a287ca6d6f60e", "805a10c2fd794ff692e8ebeefd65f2eb", "3730f843399d4ba48e98383563283e94", "f855aced7ca2485ea720604359deaa18", "6d728366de1a4bacb1ba1939c5e0146f", "c57c7cf35bf04f3bb3b2b0d8ce7feb31", "2113970533604749a808fa1b98269b0e", "fc15c6d6eb3049a3b8542b332dd8a3f2", "b6c06e70d7ed48f1840086489300cb3d", "49504531deaf4449938bea751d1ec4e7", "7dfe540a75864cf390b3bed20ab1dcd9", "c81d20fe47ce4b7594427830d71504d7", "a14542c8431c48b48a614cfd0d41f03c", "856f3dcf949741acb394f252186a1d7e", "865bae11c917492a9a1ef7286a493bd5", "2561b7a7c1694f229d30d2b1eeb14b2f", "96bd12acd63a4232b2f4ac159cc4a768", "746c8a2fd0ee458bad85cc75ac333e43", "ce6de6f9ddde4a6d8094a2b96eac3a4e", "62ceac028f144120a24e75afbaccf306", "2f953abe023240558580dcff3f0c033e", "c2b42681b8bb47e3895d6105240c5812", "ae58b3f129644729aa2f890b1712fc6e", "74a968ce2dc34189b7dfbb39276f5138", "d4f5b19f75e246df9c688f625792e8ba", "5cd7a5bbab5f4f2daa53d984568ea630", "7288f4fd6bb44089baef9f20f50e0e04", "57ff7d85e56f49239d1eeedd43b88d22", "4d5cbca8d5e647669faeea32de761dcd", "a6ce291698ad460394433a49000c1d25", "aa54b2a9b43848b0902d135beffb806b", "8cd2073cd3304b2ca4b997127aa88bd1", "f51c7f18977447e2bca36e1da3e1be4f", "b8c29c1accf4479da91d393dd59ca82b", "5fa269b8704e4509bf8cd918b657ae23", "ac91fc78fb04436b80a0a2cbe4380c2b", "5db87509bf56435eafaea4cf1153a5d3", "840981ef419a48918cf000f89807890e", "2c27ab87b28946e39681a29cc6f14ea9", "c4ab408eb1344da0bb15a9a6760818e4", "d54e8d69575f49eb977da64abc5ceb0c", "548299d96190406391cab49ad29cc5d6", "c41a9d785e884ab0a58117d17ac7d228", "256c56849e8545018514fd52f1247501", "041944011c204558a4315c27ec5dabce", "1247ee7aeac5406e93e22313d54ef54a", "ed2736d862a94d8f9db9ba6037016071", "ddc333530c13446a91cf332846bfa22f", "3d2325879cfa48048e81c44e2c2444d5", "7573358a50bd446486b4c5528a298fae", "663c338f84c94b63bc05b0fb6835d99f", "3185bd8ecbde4f26b8ed0f92cf79e14f", "ddc36fdbdd634dc489f658bead61e7ee", "d7e33c29d410414eb452d121edd9920e", "d3511ce1754b41969e5c36a5b33ac466", "9225dd50bcea435289fa2eba64087076", "a23144916f02459b966b9830f0a1d64c", "c9b718882fec4254bce1f33fa9373921", "f27905d0073e493cb9dcd174c0f15e35", "0f634721a70d4e749cb616e55ab747b3", "e1f62cbd805d4b8aa9aba7e345c21c82", "893d8cb7857147ba82ac86d140f69a5c", "4884bf82f5814e049c47cf6d496aab08", "432fc8277ebc492f91d6b46ed073ccb4", "c423a3cc0b504828b11077c77268ed92", "668dd47eec9942bcad0af209772cf8e6", "a618f430e06645eb9c95bf16bb6ea59a", "bba20d9bf1974a7f8d1ebcb9f5c4cd49", "5cec06ccf9074e019ea1f7daf17a0319", "faf24b3ed994422f8dd806ae0cc30531", "79f7929f423d493caf86b548e91e8a42", "23cb4d44c4534bae80357c8a9f6b86ba", "4fa82dbe7d3a4b3996b1fa1cf9c23498", "6eee224b244a4cafb3fc0f3b16eb0d03", "73344836205f4569a4113c24985bd5b1", "86dff0987bf040da99c8f2846da26d86", "63f8ad255d2147128bdc26f47fdf2528"]} id="AQ_HCYruWIHU" outputId="e9baadaf-e202-413d-87f0-74c730d78408"
import transformers
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# %% [markdown] id="Duak7T_B3VpJ"
# ## Share adapters on the 🤗 Hub

# %% colab={"base_uri": "https://localhost:8080/", "height": 331, "referenced_widgets": ["262f01ffc5824b5faa8a61afac12ff67", "c1af10b599da43a3a848f3ba816d7acc", "be673691e713472980fa1132465714b4", "01f8b90f0f184dfdb92c0c3bffb28b0f", "9ad2bd0e92174d339a3a91a38253a180", "bff17ef6aabd4aa681bfd5ad64b808d9", "55633800b60a4336abea6a4adfcfdec1", "73d5c6b4034d49b392b103d889bfb3b4", "a55954b8d7bf4057b0d7aa6a1cb9e91a", "4da42eb3846f423d88e2a6462a0cfce8", "dab39ef354a84be3b37b6f151f9d9b9d", "b88b03326f464c96a5656eef774e36d5", "4eccb670e98043b3b2702821a3060ece", "a333501a50df4b9fa9546d8d965e0dc3", "1f173cb95c5c44f4b32f6cfe10ee3b03", "52c8a7e673f24276a07042388a13b58f", "a0f323ccfbc14fc4b7a5e7046b221ce3"]} id="DpYr24pR8T_0" outputId="20186456-1bd4-4655-b2f2-8f24f9f37fcc"
from huggingface_hub import notebook_login

notebook_login()

# %% colab={"base_uri": "https://localhost:8080/", "height": 133, "referenced_widgets": ["3dbe077ed0c34e4eb1628418138ccbc6", "a579cb7804774ca3aa9efd800d1af57e", "9f44e3175835470aba43e239661037b2", "549bd12e3af64256a7534903688835a8", "a18d9beb34b848ff8cc541d2cb290c4c", "876cb42184f54b749c442163290c2c45", "43b14e0e1263499dbd592b280cff21b0", "7164feaf360d4300a5083f95064b144a", "bd7547126a874a45b608afed7ab2958b", "26e4b6b94e4540728c59df8e481ce43d", "9d1b77500f1c45308d4791a9c443e307", "b2693135b6954d35afb3120f3caf4000", "aeca2429ee48450a814515cb06acbc3e", "434c09950be04d728cd7ca8d6c134dc6", "34f58c132d2e4249b1e62a0b57c85999", "d148369be26a43949257790cb202728d", "0ecf58c5cbcd40908595fccddff1c6d4", "fc2314656a2745eb921f636cc3451381", "7817f8669b7f449fadf02d7145fa89e2", "06e012eea9714ed589344a362b7421bb", "504e9e5ced0348cc87aafae0c1c372eb", "5b538e8389fb4574a5dfdc554624e3c8"]} id="VxB6UV5XAvvP" outputId="c3b0133b-f5b1-4283-8367-f06524bea46c"
model.push_to_hub("ybelkada/opt-6.7b-lora", use_auth_token=True)

# %% [markdown] id="S65GcxNGA9kz"
# ## Load adapters from the Hub
#
# You can also directly load adapters from the Hub using the commands below:

# %% colab={"base_uri": "https://localhost:8080/", "height": 517, "referenced_widgets": ["ff2454cf69b346fea70070522cf93689", "87b36eb4ad3b4047a32c7d67a5aabc5e", "8bf9fd4bd28e4bc1b5895bc9315e727e", "01845549768a4db580b5555809e83342", "a490025901df478f93be1f19c1be4b09", "4f0c97b8555942aea6a8003228c07427", "80aea5c9422648ef91292d75fb46c8dc", "8af46fc543bf46a68ac78c2e8e7e9ddc", "b7b8897c988445ae8f9db9b928674477", "4bc518f16b7644e5a11bda66cc285d89", "441a953149814848a24edc9398a2ef63", "75913676a5df43fbbfe744b8882188df", "1e94c570880449d0a2763bc1eba09057", "d72bd17e161442b0979dceaaef66d82c", "2396a162f70d41a384afd1fcb2f78d11", "39f1a52884e3440c8b521e493d6e1a53", "528b93d088054955906282b1c9f93a17", "99b9d36317a34143acf982fa3b089fda", "4f592d7632fd440aac0bf97ceed2de75", "1b515483e5884479b2101127c16321d4", "ce8d4bac782949579a2e52864455d9de", "523deba849044669bfe8c959750708fc"]} id="hsD1VKqeA62Z" outputId="83ecbf31-3f84-4b8e-d667-10317f7adcec"
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

peft_model_id = "ybelkada/opt-6.7b-lora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# %% [markdown] id="MHYljmTjj5wX"
# ## Inference
#
# You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference as you would do it usually in `transformers`.

# %% colab={"base_uri": "https://localhost:8080/"} id="MDqJWba-tpnv" outputId="dd65644f-6921-46e1-920c-400bd7d888cf"
batch = tokenizer("Two things are infinite: ", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# %% [markdown] id="HZL8ZbcJCHoy"
# As you can see by fine-tuning for few steps we have almost recovered the quote from Albert Einstein that is present in the [training data](https://huggingface.co/datasets/Abirate/english_quotes).
