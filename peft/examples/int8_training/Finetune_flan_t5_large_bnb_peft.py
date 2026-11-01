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

# %% [markdown] id="lw1cWgq-DI5k"
# # Fine-tune FLAN-T5 using `bitsandbytes`, `peft` & `transformers` 🤗 

# %% [markdown] id="kBFPA3-aDT7H"
# In this notebook we will see how to properly use `peft` , `transformers` & `bitsandbytes` to fine-tune `flan-t5-large` in a google colab!
#
# We will finetune the model on [`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank) dataset, that consists of pairs of text-labels to classify financial-related sentences, if they are either `positive`, `neutral` or `negative`.
#
# Note that you could use the same notebook to fine-tune `flan-t5-xl` as well, but you would need to shard the models first to avoid CPU RAM issues on Google Colab, check [these weights](https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16).

# %% [markdown] id="ShAuuHCDDkvk"
# ## Install requirements

# %% colab={"base_uri": "https://localhost:8080/"} id="DRQ4ZrJTDkSy" outputId="31b108ee-a34c-4a1f-a970-6fa1809b64c5"
# !pip install -q bitsandbytes datasets accelerate
# !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main

# %% [markdown] id="QBdCIrizDxFw"
# ## Import model and tokenizer

# %% colab={"base_uri": "https://localhost:8080/", "height": 350, "referenced_widgets": ["59782bac41834abbbd6cbd6614c2e57b", "8252a05cb70b46ec8b0480062ea1cb71", "b77afc7c1f184de0970feb2df8ac5285", "23e494a586cf47d1afa5e619abfcdbba", "bdfb4a04e48246a4b0890f52d6dd424b", "8669a890db6c456cbc3ada28976be30b", "9e5afa2048c74754816b34a34171fcb0", "5bfebc75ec424c6cb41b33d210d28d2b", "ef2fa44d0105457c9aed3812633dd329", "5a6b38b8fc1345e8bd4e8f3aaea546c9", "57d4ca686af34205aca630b1c61d4aea", "dbdb787728184aa1a6906f96c5e6f929", "e3be963920c84c7fbe7e0bc61b8e778d", "1275c5a5c88b435a897f88a19c54a0a5", "81d2f0953e104fc1ad57295819b6b689", "61ab054f49884b1fadf529a39ccc37dc", "924e6a8308fc47af929aca1987a12f09", "c1757a5b684f4496a4b0e3db544bf44b", "a3bb3f44c1754082a4f5169431c5b760", "a0186b2194df4a0a9cd1ac49054d68da", "bcda86e43607436583f1fbfee08a9786", "55cf3bcee7c745948b39eea5f65fc62b", "aa2c51ad05c14a02a13e5c047779fc05", "2ce6779fa5904471945fa5738510af64", "bdff3b35dcdf49e5ba2c5c2498773cb7", "6994741f3113493b9d5bba278b8732f5", "808a41f78a7c4ae0b6aafee59c6234ae", "c7771479ae4e4efab744fad6da586fd3", "2d7cb8f5477244ce82e7cec5a9f7ee39", "df8fbdbe9bc341e3a39a7bda99b70be2", "179a912bbd1e454eba503782b675efa8", "232fe29739b74dabb3b558c33835eb32", "978bbbf33d304588af971d22bb2a3690", "1b839f5b472e44148c9e02a12550fade", "b560fc36ee8f424f9590e04a042046fe", "0845ceae96ca4c95ad4ebebac46b691f", "e1769695dffd4ebeb79a63ff4812fa9e", "ec73524ed7f14ea0b67f07d72eada173", "f2a36b126c1b41848e61b0c581ff8c4b", "6a40d6535f9e4b5b9c9283a1cd67687a", "03abf3bb217e4622b31cd869ed11aa00", "24f9db6df6d54ad6bac4891b65184799", "f66e179caa8b4393bed19a0488821c47", "cfc7aa04c11d408c9c12cdbd9cff4bb5", "65aacfc888014bdea55253428b2569c0", "82849bb4d5da452e87a18ca749ce5d7b", "2d2fc7abb9fb411c810b2fbeb54d67cc", "96e2d208830f48cd821be7e59643c93e", "8d79b7d0c3cb4f8d99fb20941c35856f", "2b4d68606bdf4758b812f5a8057af595", "f9620e01cd6749f88b722a42ff68c502", "06c3d8c44fbe4f0886cf397d9a3c4b81", "d72e8b3419f240f2bdce253cce9d24e3", "2fe0a2fa22a0498da983ec38150216e6", "f3784e85cef34bdba64b611a1f5883e4", "d7d9e2e2090d4226ad89e5ba9cec33df", "5cc620a232bf4d418c3fc882f4c1cd0c", "f742450a607c4ed0bff98ac9b7685d40", "e87b05e685b040f7a99450bfbab72433", "e1c8e6f843604161bbb6cbd269488469", "34333da7eb0f49a6b83df76e35dcdd93", "c708031a279e4e55ac7833e6697f93bd", "e293930c8e2c4eadbda53005e21ec450", "945ac449c2e84fd6b5a7805b017343f2", "06cea508d7504b228f6cebc66742d200", "9d0500a0f5f74be39e5edfbbcd7a64fc", "0363c4d4dc1449148b09061763ea119a", "f4ff06e2c48d4e58abe64cb7f41dd886", "7fb7e3e2c75d4d03a98e581d4ead0f00", "b8087054f46c44cab9bd62fa23fbf9de", "85ae7ed1ec244a89aeb9f4552c2c9462", "4f57cfa7cb3b4199babf82dc9d93b074", "053de11f995247f6b851909a6a8dfc16", "43d7a9b421be430286b5eb8441d6d465", "ebc26228160046c48279d71770c928d8", "5192bc282c4847cb9df8365fc22a6cc2", "608e9f7a14054573b9bd07f0f74b6345"]} id="dd3c5acc" outputId="fab398b4-dace-4f44-89e5-687ec74b18eb"
# Select CUDA device index
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

model_name = "google/flan-t5-large"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% [markdown] id="VwcHieQzD_dl"
# ## Prepare model for training

# %% [markdown] id="4o3ePxrjEDzv"
# Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_kbit_training` that will: 
# - Casts all the non `int8` modules to full precision (`fp32`) for stability
# - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
# - Enable gradient checkpointing for more memory-efficient training

# %% id="1629ebcb"
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# %% [markdown] id="iCpAgawAEieu"
# ## Load your `PeftModel` 
#
# Here we will use LoRA (Low-Rank Adaptators) to train our model

# %% colab={"base_uri": "https://localhost:8080/"} id="17566ae3" outputId="66cbd0f3-815d-4d68-c0a3-6b2e5f46b021"
from peft import LoraConfig, get_peft_model, TaskType


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


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)


model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

# %% [markdown] id="mGkwIgNXyS7U"
# As you can see, here we are only training 0.6% of the parameters of the model! This is a huge memory gain that will enable us to fine-tune the model without any memory issue.

# %% [markdown] id="HsG0x6Z7FwjZ"
# ## Load and process data
#
# Here we will use [`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank) dataset to fine-tune our model on sentiment classification on financial sentences. We will load the split `sentences_allagree`, which corresponds according to the model card to the split where there is a 100% annotator agreement.

# %% colab={"base_uri": "https://localhost:8080/", "height": 297, "referenced_widgets": ["ce0213e9d6aa45c5a9ac9954fbe15f62", "7cfda0921e5a4f378e90e057447f3b3d", "aaa1477cfabb4767b755e902d3b99e61", "7b6bddd4ca51495dbc2fceba7c50706f", "6317d49813234f5b9103b249cf648c2c", "10aa4e3aca57438ea7af97b60208ac81", "f14657da8e1e4298a96e3885eb4eee93", "a38c0fedf90a4f3cbb4680b5f85bbf2f", "c6a28dcd88c1487ab17aef6946ada876", "2c47350dd2164d4a8164be2ec6d16391", "8fceec1018574003884e082b2a5c23bf", "f2dc5e8a31c348358aca916274899e8b", "2bfb7c240e154769a0d58a3ceaa20212", "8f339c9070f046dab46ebc35c1cc2dba", "34db70b6e6ec475699fd23a2d6c3a973", "e24115bb662c428e89c2c4421915e632", "8b41d2e9f7424dc898446e7f428dc757", "8035fd17e29a48d7b415c531607216a6", "d618ec6be7d14a239b3bc74172616bf2", "3c8349539946412a93a51d9087306ea4", "645e26fd9b3f4919a43eca3592475324", "83f196eb5d9549cda4d48008fa7b1386", "94437f56e5a44fa3bb08c9d798b2eaeb", "0f091d25adf34ade835b094eb5b952a3", "9d599c2a4d9f4f2db1e4b3183c18eb94", "aa91ad725da147bc8cab70f931d82672", "fb6877c376e0430296b2746513f60931", "2de762c8d96b48c7bbd72fad566611b2", "8ef94158b0584f0eb55582bf8b6594c6", "bf7a49e0e4a64df6b1b1c66e5e73c3a6", "d27ff6c2869242b98564b0e03d68b413", "8a2bd1b4d9ba47ef9e77048e3d2d1e83", "dbd908538859410f9c20536fe5acb328", "cce112d791dd4b748908756e785ab555", "be11f6865f6c41b5a57b2b7f4a85e14c", "c31cffaa6934407399856235a2f3af54", "bca79be79b6d4a68b148255bba86ea96", "5093700dd3a14cc1a283d18a4a0e17a7", "85e5c1a9b7ac4e6e884213a636d0aaa1", "216e5237b31944cbab006d9761ade0a1", "a4d6de73a37148bf9303a273d13cd091", "f79eeece093f4b0e9de6dbc346a3fa19", "74e88bd01bf14e0e9f772f993c92eb77", "401bd48c5b2d48eb86a1499912ee2b44", "41d8dbe04d0c4ec4a9d19662f9e11920", "1bf4434ccea34cbc803d88f37aff2065", "07f5dcd5a6e24d819e0ce9e54a5bfafa", "b8944b7027d449b4a7fc752978f463b1", "baf53867f52046c182a2b1755f02e136", "808cea6c94264f0c9990d6dbcf538419", "96e4e44a789a46ce8239b260bf6e3dc8", "2f20ea2be56a4992827f960d8c6d7b7f", "2871be167d4f44859eb2bc0baf0788f6", "390b88f67b84451999b0845483905144", "be9c243b74d944eb82ca1fe4ada6721d", "26e1bfb23d3a4797a8dbb9c4ed2aeb22", "c0deb08457be4a3ebb3947e33f7ce1df", "6a85e6040bf440569a23495d67a66de1", "30c87e87c4b845b8bb27507596f4d18c", "f061a6deaa73484aa04f219bba6a4329", "cba58e0b316b439ab035b917a40c630c", "de6718209a7a42b0809e97fcd97e09ed", "66535de16cca41cd9ba44b6de40c4e6a", "51335afbfc02480fbc8d2c6200f3e18e", "dd53a486f7b5403a81e2be89cbbda719", "5626348c213b48faa61651c08cd1bb24", "283bdd8a4daf4f5ca1ec6e9ed19c46ec", "9d68afcb8e26420cb91ea1eb872c80c4", "40171f7e3e1f46b4955548bbb58cac6c", "b8bb0aed01d04e8dad560df1b051e1e4", "da99eed13d524b8fb95dbc563eb2d044", "a72073cfb8b4422a98ca581c4e5d18b8", "3a5712c976b04af0975804b34344dfcf", "356cb75b5f4247d9be9d4d2aa15f2dc1", "64670ef7321140de89ca726b5eeae337", "93fe5a8fafbc44b496309d1a8da77ac5", "a9effc13b52044a5bc0d6a2a1088396f", "8d35a041dbfb4747aea427e76890551a", "74efd6bfe71e4dc599a7fc76574ff154", "1ed1bfefa6534085869130ea533ff4b1", "fd08c4fbe5d84dd893d87a5e2f2d082d", "87ce7c58b18146f3ac73970d7f8079ac", "39dd984ff238456e9e485c077f9e87a8", "28e24fc8410d486c904f249579d31b8b", "6b05e37f42b04701a97bab91ae92c2dd", "2ec84788d5dd4a04a24ccf66dad457ad", "6f30253108fb4dce9c3de029457ef6f1", "1fa2a7e3ff3c4c99ab95e96a28624846"]} id="242cdfae" outputId="3122a25f-4299-4c3e-cd93-3a507f9f0d91"
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

# %% [markdown] id="qzwyi-Z9yzRF"
# Let's also apply some pre-processing of the input data, the labels needs to be pre-processed, the tokens corresponding to `pad_token_id` needs to be set to `-100` so that the `CrossEntropy` loss associated with the model will correctly ignore these tokens.

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["b718fba0f1514025a0ca22e7f780a2fc", "0271e1cc4e2d43c69d4959e46eddec9a", "601fb3752e134641b28da908d4e7b65a", "e002207d6982491cbef196f25fc891f8", "a4857f97132a41acbe4535b03cd8d94a", "d5b57d3c74d14e5d80d1ef634c103a40", "3786a8205e004fe9a5e069acae422410", "05cbd69de3664b82b95e06f72f12a55e", "c149c1c53e9d44008a86944ef8c261c5", "1d11cb45c5cb472aa86722e4dbb8c085", "34142a8e97594931b316970911679e55", "f48454eadbfb4953b719bdf44555c90e", "3c5affff513341b29e6a2c1c90bfe334", "0bbeca449a814d95bec438a9141b2b6b", "a10078c15aae4ec6a849f1b58c6b1cc2", "06e8fd84d6224e5096088d66aad71961", "456429d05857411f8b78f69e8860bf58", "f00b73eb32374c33882c1bfc49822e44", "bc24304c057d4b5898e832818de55caa", "40b951e3a06348a39c7c778aa12f8385", "223848818aff4af1ab5d5e14271408e3", "4b4b31109a9746e88ffa9b47bab00e53"]} id="6b7ea44c" outputId="a27a9252-eb13-48ca-cd9e-4953c8bcb75d"
# data preprocessing
text_column = "sentence"
label_column = "text_label"
max_length = 128


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
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

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

# %% [markdown] id="bcNTdVypGEPb"
# ## Train our model! 
#
# Let's now train our model, run the cells below.
# Note that for T5 since some layers are kept in `float32` for stability purposes there is no need to call autocast on the trainer.

# %% colab={"base_uri": "https://localhost:8080/"} id="69c756ac" outputId="f0d605b1-3b5d-4e22-e108-819edc7b0d52"
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "temp",
    eval_strategy="epoch",
    learning_rate=1e-3,
    gradient_accumulation_steps=1,
    auto_find_batch_size=True,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=8,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# %% colab={"base_uri": "https://localhost:8080/", "height": 659} id="ab52b651" outputId="2da171de-e59a-4945-93cb-5704681e84c1"
trainer.train()

# %% [markdown] id="r98VtofiGXtO"
# ## Qualitatively test our model

# %% [markdown] id="NIm7z3UNzGPP"
# Let's have a quick qualitative evaluation of the model, by taking a sample from the dataset that corresponds to a positive label. Run your generation similarly as you were running your model from `transformers`:

# %% colab={"base_uri": "https://localhost:8080/"} id="c95d6173" outputId="ed03a1dc-597a-4053-99d6-eca2cc6da253"
model.eval()
input_text = "In January-September 2009 , the Group 's net interest income increased to EUR 112.4 mn from EUR 74.3 mn in January-September 2008 ."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

print("input sentence: ", input_text)
print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# %% [markdown] id="9QqBlwzoGZ3f"
# ## Share your adapters on 🤗 Hub

# %% [markdown] id="NT-C8SjcKqUx"
# Once you have trained your adapter, you can easily share it on the Hub using the method `push_to_hub` . Note that only the adapter weights and config will be pushed

# %% colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["5bb29f3102954b06bec825f6b3a7aaa7", "90516032070a40979181d1d27db10c4f", "4b7dc0fb222b4e2a9bb2ef2501e9fd30", "06069855ef82484f9985e4619095dbe8", "1ece69c53e37413caad8db70d9160ad5", "7ce90db727ea47cc9344176858a2225b", "64f2b70b63cd4e7eb9e22ac2de5589c9", "57dea1b3e04142bb91868a474774d86a", "8ac43334e0ad4a78acda3b876fead058", "04da98e400514cf2847d172916cd0081", "4dbe49547fe94010ad5a30818cfc35bc", "99091ca45c1b4809ba0a1b01af85f528", "06ec124c3dac4fe6b152fb812d20d86d", "7561c47a97444666816422a0418e1675", "45ab5d7049e34dfd8a067643ae887a31", "434e308cac5847f0bee431c7dbb4c04a", "dd993a4a7cdf40448098544c95468a10"]} id="bcbfa1f9" outputId="91ef770e-9fc4-4eb2-b02b-24e635101f97"
from huggingface_hub import notebook_login

notebook_login()

# %% colab={"base_uri": "https://localhost:8080/", "height": 152, "referenced_widgets": ["00f7d043cb184d69b828c204dac2c0ab", "1e409cd3d3a04b558d989d63f0b3b5f7", "d48cbb38078b456fab1634bec5b0a1ba", "9030744dbca9427ba8a036a76b5c8bf7", "53dd4444c0e14e16a912532898b32d92", "5287ac638c22412ab91c55f3316c9b63", "a43ddb478f044f17adbcfae841ec2114", "45141234ce584f208a9d301faadf75d2", "f62ffbdc24734b999f36058d9edca81f", "972ddebd536d4685bfc3c7c13e5bd8be", "64156e2c54b44fb9aec661d9b57da962", "050de732f51f4af8bb41ab3cad0090a4", "7960ed3beb2a429ba2aca1c6ed032f64", "726a2eedc7434210bc5aa4d0a772b313", "07bf5d621cf944258aaf13954669df56", "93a0896ca66b4111bc4cabe6e1278440", "cfc78731f7d543ce8529cc254d92ddf5", "eacf8e9ed6e847faae2b8ecab283ddc4", "bef1971d92e6479696e3f9a27a757b8a", "821f2f296acb40ae9bb40fc3faf4103d", "11efd993475a4f2aabe7df605bab04dd", "406e4d8561f64d2a94d93a606d02d7d3"]} id="rFKJ4vHNGkJw" outputId="07425379-64ad-47e8-ba8f-8d9dc26252b6"
model.push_to_hub("ybelkada/flan-t5-large-financial-phrasebank-lora", use_auth_token=True)

# %% [markdown] id="xHuDmbCYJ89f"
# ## Load your adapter from the Hub

# %% [markdown] id="ANFo6DdfKlU3"
# You can load the model together with the adapter with few lines of code! Check the snippet below to load the adapter from the Hub and run the example evaluation!

# %% id="j097aaPWJ-9u"
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

peft_model_id = "ybelkada/flan-t5-large-financial-phrasebank-lora"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# %% colab={"base_uri": "https://localhost:8080/"} id="jmjwWYt0KI_I" outputId="466e79b6-c5a8-4a03-9a17-eeb2bc27e9ac"
model.eval()
input_text = "In January-September 2009 , the Group 's net interest income increased to EUR 112.4 mn from EUR 74.3 mn in January-September 2008 ."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

print("input sentence: ", input_text)
print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
