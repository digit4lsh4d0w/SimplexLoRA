# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="CV_gQs58bsvM"
# # Fine-tuning [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) Dataset using QDora (quantized Lora w/ use_dora=True) on T4 Free Colab GPU.

# %% colab={"base_uri": "https://localhost:8080/"} id="FuXIFTFapAMI" outputId="b95d8260-65bd-405f-f1e2-8d353aa46814"
# Install the libraries
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q datasets

# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["8cc86330c2af436c9af314e8c04c8c2b", "e25f9ca445b14e3f8397779df071dfb4", "8365680c634a44aa880317e36fa5e46e", "0c4ac7c3db0b431397cc812f7c9e785c", "c84f542c863043dea8a3675fa153e78d", "b3b3f4ddd4ed4d938c923887939a0440", "35186465f87341f683affb9399661540", "791df472db174df69b8c9f0e200af254", "6bb9c7182d2a464ea21809e59043562a", "31c574113731403b88edc5bb0798bc6d", "3b8bc5b9392e45758813a1db9db824a9", "90661b333d6f496ca606b3046622660e", "5f551f9b217e44cf8b5433f314b3844b", "d2d81cc8296c4b10bf80b86c0a3302d3", "7e3a386e672f4748882211227b7721a9", "57f251691b4c453896b2508c431dfc2f", "4bdb196cd1494f809829651ec5b6cbf8", "7cd50bcc8fcc4b83abcda6d3604bd4cc", "7a00aa4a97a34da39cc052c6926dbe13", "14c73d88df9e46e3bbb6690fdb48ad07", "0e2beab611114239b6ee48a3cbb09c49", "006b78b5191b4fb888d98bdf6c20ec1e", "5f6ffa1d929443a5bd9c7c550f0690f0", "668a7f88506148a9ba2b48920afc028f", "57b0096985ab44aea342e52795c4f999", "a4c404e420cc4ce781ce569f9ab3f987", "ee4e4af964ec4dd597cb04a90f0697f9", "974e3687f18a4e1a975969b880d086aa", "93a50117ece543d4857ba02505dc4514", "71a3a56edbdb45669d382fef4b097e1b", "53f287d4927541d08e2ae7d4d0b3c396", "afa442ab223b46cb82569438c0047823"]} id="wAAPv5CRmg7e" outputId="687f979a-04c1-4160-d71c-4de8ecdb07d9"
# Required when training models/data that are gated on HuggingFace, and required for pushing models to HuggingFace
from huggingface_hub import notebook_login

notebook_login()

# %% [markdown] id="MJ-5idQwzvg-"
# ### Loading the model and it's tokenizer in quantized setup!
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 546, "referenced_widgets": ["5924b266e95a42039634a334ff561a82", "eadeec171e7b4c0f9e26964f031cfb71", "feae525923d5407bb69a922954c474f7", "00371a48e64c45cd97020a78b710e64c", "156f95b0012449e8a0c604e6e03bf35f", "de3757d6125a4c07b502dd60816bafec", "8c4d6f4eea3742289a2604e66b0c6182", "bac377ed96ae4e8db9b298bb623888ec", "02d6cc4c2717434c895798601bda7c86", "3f9fa554747743f8a86b40a4f7530617", "ffe561df8772443ebf40a3b8b656079f", "800e9453214848b69bc4c6ca2d5e8f79", "6b0ec8d5f7294d44a5fa15d8ef12471e", "7d3a7be9ed6f48988a2c4a1a4a2271cf", "c40f583823574e40b6b29d4914143c0e", "a4368e6da8f046aaa32f3152b7d333d1", "8ea89e52123643268857285e0e1db1c0", "a8514e34378d47a28fbf0831a14ede8f", "f86b969ef69b48119619e1a424b50460", "7725e9d443e249ada02e5ac7056d00db", "f81756eb9e554899b0778311f2c407c4", "3b614b9712874fac990d2c557b0791a6", "43d12a98d90a4bf7a96c033172c646e2", "c35b16156253402f90a432f3f07c2e0a", "1ae1d2702da5483a85504f59939ffa39", "1b2abf90003e4165a3293acd6a5ea9ff", "3e45aea9f7444a4db885c4cca4c9c4ff", "6b6ed29053ec4aaa8fc5526a35f17c2b", "e69cd88ccbae4bb7b238fa112a60f0f9", "8cd63d3908e4411c9fcb42bc32c8dd16", "64624b26145b42db82f7afc36c32e117", "cffdf12fbe97462ab74e88ccca943aeb", "bcaf4c81ba9d437bb6223dbb22d011ed", "8800c351b6da450eace0c3890d36c8d7", "7037c32dfce84e70ac86537dbbc6a495", "31c10fa464e24f97b379675a204a09b5", "6149752353fe4f9cbb7b26bcc25199a9", "0b145e421f4840f2872c29256b49f168", "0f4c664612364dc89acf78eb1c740980", "0f60f9aa76b941809e013ffcae83604a", "1d27ab2bc6ae463a806292b68b7891f8", "2a57bb48e1c6475abba242994a79d44a", "6cb8065803724d80b82b06dc95ded91e", "8301c6302df54bbc9f15295f11cec208", "0aad2d9d1cba40cbb64308ede3242ed7", "0711e28e06a440c2a241acbc1f90d1e8", "77704d2e27e94cd3a0c5f6b5ceeffd1c", "3b82b8d41b134bec9bd77ed8d4f00eb4", "25fef90e209f4b14a73f3e39d226d913", "f4ca7b63d7d749ff83a848e250f03ec1", "8c149bc655a34fe5b91853c66db458a9", "0f54e8fda93144f6a95493e6ec535e9d", "880124db7dc04aaea09edd75e1ec7921", "14bf612f6ad7416c8ddd6085c72eee0e", "f7e59b47f9b74523843f37268212d566", "a34b3fd5859a441f89cbe7f6e6df9da9", "8f5b8c513b164dab9e0892422163c483", "5b2a671976fa446db408d58a215b8249", "c934919f617447cfb9226929e7a68d79", "124a70bfad434c5c946f611c04a91c8f", "cd11fb7d54bb43ae821f2272d075a1b3", "fbc6a2834c5442fbb6667f1b3612bb5b", "63899ac621ff4e9cb8e215d5ab63bef8", "6b2b59d2b62b4f7da8c60ff783138397", "c27e8ce031884a90b41d8220b1870bc4", "88024cd312ee42c2925ebfbe52077780", "cf6d1be81b6c4ffc81ce8fdabfc5ad28", "07e0aed682fd4cc88fa75c0592dc04a7", "67b4473eb8a44a96ba34983762ab38fa", "c7e06fd82f7f4f9fb81c68e8758f2de1", "3d9d8278667d496aaea1eaaa4d24ae93", "47944ad8cadf4a57b170193c46d4389c", "db05b25cb38140bdb21e6f3b7fde7e66", "6f474268da0f4337a2ccecc1ca2098a1", "c2ceccfdb59b4336a24003cd6bc2403d", "a8999d04e4114693bb6be358bdbe9b83", "2540d57e3bf545e3812da1ee72b85fc8", "c56d8289513441688f9bc5f4b52d60a0", "a53b4776f95f4dd38197193e6c5f649e", "b4ba435f6d1c448f99b533bc6df32e76", "42eb041021214110a860924d28d73409", "17c797e08bd2493fa685918129415309", "aea74071600f483b9e6de1a61743c03a", "21bf14b771c14d2dab9e98a326302e14", "35c2c635c2024bcda3265bf95d330f63", "ec014d847e394a309b6a82c30a6fdfc5", "4c369386ba5f4862b11a50e50130663b", "bbdf3bb657e64fc2b0a90e78e8886480", "d7ef74cf4a914ad38a69c84c34fff393", "8d0f1d547c384094b10aa00a3ede3c06", "9bdebf06b6874bbb88404f4ad14e1dbc", "a2249f364b914662b54045a1f8d6dfd1", "7504986b8d8d4d0da58ad79e80a81948", "082b6990ce5e4812adc0ad6a7b376dac", "610e1ddfb7a44d51a54ebea6dad3a5f0", "5f60910d1e744432bdf87518f0f45874", "2ab86b3fbd49488bb02f8205a572e752", "9f13437a44b8434b9cc3afab998e8d3c", "8a7c82dcbd414b24b67ccfbc562b2e38", "d1508f5cde9a43d8abc26dd2d0c34dbd", "9a0b012915c54abeb100f466fa99d303", "99529129d7f0435da0fdcfc9803a2f11", "5b56ac3009714a5a84dd8749db4a7bce", "546b76a22f1046cd856a8fa2f9ff2d9f", "fffbf696c07744fc8e3d81ab51dc9c90", "a153cc3ca0cc45c18a941bd57e363ec3", "5a8ac674153248999007a713299b2644", "1f82a5685eef4b47a2dbf7618362907c", "51b3af446ace409dbcdf5de499552061", "9a12124915994b70a71ebd64b99e93e9", "57f87d4780634d36ae8159d987c22993", "58ea619f81bf42ddb8b166db3deb0e86", "8bb83ae3229e4f38b1733f92f536fad0", "5c0104210ee34ca8a072ee5121f424a1", "34e381adbd9242759b57f2a305c5d2e3", "e4e1a4338c5e46b3ba5a3bb960da7107", "9a5072b8d16d4a1eb0652da61bda0ac8", "1f75d85e6c7e4eb6a91b03f0c8adb644", "9da33f07ea354b5798e85298e132b017", "604582e8cbff4dc9876551a3307b5b77", "a98165ee656643ad85ac9ea1447cc775", "5cf4a57d21a545029b6448258a5ebd84", "0d2ae3466a3447c58e23ccd2b3733deb", "ba7f32c41f9247ec9d4c40e6396b55a9", "ea1bdb5f2da64332960bccd967a84b4a", "b08631e4cffa445c912da0c8eac2ef23", "921a1a037f7b47f8b57d1da8192a437a", "892ff4e2f0e44c23bc5c2be7547cf0bd", "4c8e98294bd240a6869cb199caee66e1", "4e1f5423311b4dc0930c21c9ad5a88f5", "347540dc03d34e65b7ffbb0f5fc569aa", "7243d8e2e1cc4043a2ee310eabd0ac09"]} id="E0Nl5mWL0k2T" outputId="a942d9b0-1f38-4a9b-ea20-e55bd7593920"
# setting up the config for 4-bit quantization of Qlora
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})

# %% id="Xpx2Fq-icX56"
# print(model)

# %% [markdown] id="Mp2gMi1ZzGET"
# #### Prepare model for PEFT fine-tuning

# %% id="a9EUEDAl0ss3"
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# %% id="gkIcwsSU01EB"
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


# %% [markdown] id="HVTAJuKyM0gX"
# ### Setup `LoraConfig`
# ### To use Dora we set the `use_dora=True`

# %% colab={"base_uri": "https://localhost:8080/"} id="Ybeyl20n3dYH" outputId="ea35ec70-13e4-4f23-9481-1b33b1b06dec"
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    use_dora=True,
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # parameters specific to llama
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# %% [markdown] id="FCc64bfnmd3j"
# ## Step 2) Fine-tuning process 💥
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 277, "referenced_widgets": ["b2a19b6092c44b20886987b30f1bf48a", "1f0efc167b3744b38ff832b71d529318", "a06b2bd0236249999adffa44e53cf80e", "23012118a7314a3f838870a2aee9ec90", "dcff079d850c423a83eb70105b816ee4", "64911f0e52e74067a1a986c5edfc7f59", "b4a274fc9e324b80bf559c4dbd05e319", "5fb4a4ef8afe4ea4af6655faea17f354", "b1a03a5e9bae46129830daeeb23bf6ff", "e297072ab5d64815b90bc89d22503378", "67fbabb9082c4241b8f937b24e0cdd03", "b1de7b283eeb41828e8093e60c83f2c4", "bb640a5c858349d29c13ce5629e72f22", "37523a6cac1047e9a261698212d47737", "661f76474252493caae8f7d6aa8f99b7", "849cdc1912aa4df4b0c721a8c63ca0f9", "5962e77eea5a4d88ba6dbc5e9f51c709", "f07c8a6ec12f46ea9e32a2208e70bccd", "191caf3a38eb4191a35f623ce25238f9", "30b74bd2db8d40d08408013cebcd7661", "83c355e1418140a5bbad11bf0646b332", "84d6d2a6afcd423f9b609cbb2d10f00e", "51180cce01564821a170d1d4b8a9a918", "03dd6c24f6d94fe7ab85b79d6f6cbeaf", "f2ab2fa803e94328a237e84cd4ea0027", "1bd0a270c7ee409c970763398e54fc36", "e92b30d0b4234af6b5a33bff989b1b45", "cc5ce633746949ed98418cae9f68afe3", "4b1f795c4c004cacbf3660d935e52995", "5690d92586494b9187147f32fa708405", "317cda72329c4043ab0b224b46b259d3", "076357d4bb9943bdaa1d6846897786af", "7b3e136fc9e74a699497a947006f4f1d", "7e2e097c703a4a0d8556733a0739469c", "e0fd6d00f0ba4e59bdaa5779556ec4ea", "cd318c6bfc8e421a9bfcdab16be5eaa7", "4bc1fd9d480a4799954c69031c071b30", "25fc6aaf37fc49fa822df29236bf2f90", "7ac8e88f29f04b859f592a003d39836b", "0ec2643d9fd44785addb37d9ecd23989", "d10ba011d05045b18bbfeb9660e4d9d3", "e56c22f77c884caaacfafd48dfa51a55", "068eb104d5d346b1897f8cbe9860d267", "b621c6a8c0e9440fa840d75a1b1b02fc", "434fe18d50a14920b30fd2d0650297ac", "353bf45a4bbc46d6a798175f152399cb", "bfcbfe4184774fd3a8320f4f0e1baf54", "99c5c846cc5e43429905f071670b4310", "8d988c86648244788f6dc5aa0fea38fd", "740604526cc44cd58b811827d4787d96", "2558c2dd7d394ecf9fc67a69ce8fc97a", "a2a7b715b16a41a288209dee1de5d2d1", "bf77e5aaab0547f7b2beb015687552ef", "a2c543008f444cf49972a4f35c32b8e3", "bb7b8a9e42f6478f851236685a1392d6", "9cfaf17064bc49a5aded0fc53dd7cd7f", "ede66e196fa9482498f58dcdffd494a2", "a26cc7fea1a64d7bac1769d33cc74e28", "c2f24a8930be4b70b4bbbcf5d908b01d", "1f59dd66813f419999336e59a3efc56a", "026072374b7d47c194707a50f5c99099", "63ac7dafeb27446cb30aaddf4cd27c9f", "854e35df771f470b82a59f878a2a6a46", "d1cbe0ab9379453588eb438d13fd272d", "b0b7457a8b47496483da1506fb2505b3", "c7dc386d978a44ff885763ecec94dc38", "4a7c8dfd88db4bc893da2bced0560d47", "27a587021d854b79a279a510a55f9d73", "4227474e986546d1a7d31dce35a2410c", "1561cd47c42e46368677d34e7b7084cd", "8b9e961c837a464fb7a8c44756dc41e7", "d4092198673141d3b4a824d629d73f64", "d3cbfd564fe8485ba7afdb1cc54abed3", "7e51e8e0612e46b1a3403d448b39aa50", "061c45266c484ff6807dcaf4722fd73b", "04188e0cec0542818894ebc6a534fb51", "4a13203d132b45beadf140c02dc8a566"]} id="s6f4z8EYmcJ6" outputId="8ece7a1a-cf27-4602-c70b-ce4f3d7e11bf"
# Load the dataset from HF
from datasets import load_dataset

data = load_dataset("timdettmers/openassistant-guanaco")
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

# %% [markdown] id="_0MOtwf3zdZp"
# ## Training
#
# For the sake of the demo, we just ran it for 10 steps just to showcase how to use this integration with existing tools on the HF ecosystem.

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} id="jq0nX33BmfaC" outputId="94e17005-065b-48ab-9192-e3ab55c0c292"
import transformers

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="path/to/your/HF/repo",  # change it to your desired repo!
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# %% [markdown] id="Mr3rLrHwqhf6"
# ## Usage Example

# %% id="9mrOJ9l8SMHv"
model.config.use_cache = True
model.eval();

# %% colab={"base_uri": "https://localhost:8080/", "height": 122} id="AM6FNOFzqKfI" outputId="fdbe28b1-e440-45d3-bd6d-c15e744ad23d"
from transformers import GenerationConfig

max_new_tokens = 120
top_p = 0.9
temperature = 0.7
user_question = "What is the purpose of quantization in LLMs?"


prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "### Human: {user_question}"
    "### Assistant: "
)


def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        ),
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(text)
    return text


generate(model, user_question)

# %% id="T5t_gl2_f5OO"
# trainer.push_to_hub()
