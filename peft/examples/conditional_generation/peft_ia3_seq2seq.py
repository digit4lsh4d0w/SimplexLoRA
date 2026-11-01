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
#     language: python
#     name: python3
# ---

# %% id="5f93b7d1"
from transformers import AutoModelForSeq2SeqLM
import peft
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, IA3Config, TaskType
import torch
from datasets import load_dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

checkpoint_name = "financial_sentiment_analysis_ia3_v1.pt"
text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 8e-3
num_epochs = 3
batch_size = 8

# %% colab={"base_uri": "https://localhost:8080/"} id="b9e6368c" outputId="fc2888a8-4fe9-4d61-dd2d-753e751e1416"
import importlib

importlib.reload(peft)

# %% id="8d0850ac"
# creating model
peft_config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, feedforward_modules=[])

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# %% colab={"base_uri": "https://localhost:8080/"} id="e10c3831" outputId="e69c5e07-ae58-446c-8301-e99ac6b85d62"
model

# %% colab={"base_uri": "https://localhost:8080/"} id="05978e96" outputId="ea9b7d40-010f-4df0-ec64-a7146a5f8b08"
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model

# %% colab={"base_uri": "https://localhost:8080/", "height": 140, "referenced_widgets": ["bbfb7533b5ca459194e171df56b79566", "c894e8237aa34c56bb250acab1466005", "a5a126b229064812bf3dcb228118be50", "661e1b29c59a4295b594edfa4f50ff87", "1bcba805972b484d8b6aa6542c81841c", "e71f5c7f1d5d4f83b58c68d2fa310d9c", "6a567e0a1a5447519c5df10e777520cf", "7aeca19b84904906a04c12659f84ff9e", "dd4b895874ce46ceb1ad0d9bc973f98f", "b138f91be7f94008806eaf0a6988bc3f", "da14180f51ab44b48470cb9ea74d3864", "9e12d97af6124a5a8c6627708b300c1e", "faa18df899c14e9cac6721253e6c9128", "79d0ede7a5b24756aa6d34fda8c29159", "3b175b452f4347558aa3c4501cc90030", "fc4637a1b37e4e90874c71aa4271ac74", "1b8aada826a0451bb60c418b19178c8c", "a91916e02e9c424e881e45b3aa978574", "ca509bd409624c998e555c9a779b8aae", "9c890fc422954347b86d3bde7a421caf", "6f9453484ea94587a64d70f1b3a1f6e4", "48770ef159f44c01be2a75c75aecd80f", "0c561dab67914ea9b6e1aab803600551", "1e021a1954b44d69a90101a96c360661", "013e3343285f437a893bdd673fb90e22", "28802da68fb04d70b1c6bc511a04676f", "94174da0d6554be087d4527bea5b511a", "dc8ab16a1e6c4e6893c95ccd16568f9a", "72383136663448d89cf3b82b87cbb392", "5b1bdaf16cbc473081e4237f839167b9", "51f8fb45485540bb985b606d43ae04ea", "f760cd4758334ca9a43fd15612fd808b", "f60e9915d2a74ca7bc010d7684f5acf6"]} id="4ee2babf" outputId="3c413083-247d-47da-f25c-032764be0beb"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["e1e80a68a9e7429397cafc96c3c11f80", "5307864c2b1143f4b44f3f172611113e", "2e2b6c3f48974ea4aca9b7710a03379e", "aae78f9bd53348bda45967a38736cb78", "34db17e0f28d40d6abafb8acd5dda379", "8361dc2e0a834da6a0ad87f7b0cb4e1b", "56f1d9d56dd44c8aa923d09a59cb0ebc", "d93bfb366db14c2fa77b038752f69b38", "749aaa39135841f98b344ffb840df3d4", "5e5aa58adb0f48579871df33845e30b1", "c25b49b7adaa48a0a3a306aa1e0661b4", "21f582e1208a4a38ae3c0cdce87e5c14", "d9d37b8b79f24dbf837327a250a5a346", "8ba99043c350456d8623ce1d8c98f7a0", "8bf37c12d5f74f7d8dbba423a9ee3ac3", "f9d86ad7fa734f3a857505a542256a3c", "86bf02b06ed740a88015c2b944205c1e", "aef6a6be67f749908060d8038b6d3804", "664c02903cb248fb9339805bccfd6c1d", "82195b807b664a9585a76e0e50fe7609", "8621932be14f42858d841e2ac1b173e7", "71bcdb1e02144c9587879d8d815b91d4"]} id="adf9608c" outputId="3e4bc95f-1dc4-4d34-c212-6d2374359673"
# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


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

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# %% id="f733a3c6"
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# %% colab={"base_uri": "https://localhost:8080/"} id="6b3a4090" outputId="369cfce9-90f2-47a1-8653-ea1168943949"
# training and evaluation
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

# %% colab={"base_uri": "https://localhost:8080/"} id="6cafa67b" outputId="0db923d2-522c-4cb7-b694-6e2e20beae98"
# print accuracy
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{dataset['validation']['text_label'][:10]=}")

# %% id="a8de6005"
# saving model
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

# %% colab={"base_uri": "https://localhost:8080/"} id="bd20cd4c" outputId="0f25d837-80b1-476f-c897-92c3fef04fb2"
ckpt = f"{peft_model_id}/adapter_model.bin"
# !du -h $ckpt

# %% id="76c2fc29"
from peft import PeftModel, PeftConfig

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

# %% colab={"base_uri": "https://localhost:8080/"} id="37d712ce" outputId="4828819a-b640-4f6c-91e3-878b648e9a75"
model.eval()
i = 13
inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
print(dataset["validation"][text_column][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# %% id="66c65ea4"

# %% id="65e71f78"
