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

# %% [markdown] id="db4dc272-88fe-47ad-98fd-b94d4f840dca"
# # PEFT with DNA Language Models

# %% [markdown] id="d381f473-0d37-4b5b-ae9e-d2b32bab7c04"
# This notebook demonstrates how to utilize parameter-efficient fine-tuning techniques (PEFT) from the PEFT library to fine-tune a DNA Language Model (DNA-LM). The fine-tuned DNA-LM will be applied to solve a task from the nucleotide benchmark dataset. Parameter-efficient fine-tuning (PEFT) techniques are crucial for adapting large pre-trained models to specific tasks with limited computational resources.

# %% [markdown] id="23f460c3-d7e5-437f-a5e9-d029cd225bf8"
# ### 1. Import relevant libraries

# %% [markdown] id="29a35f95-738a-4f5e-88ce-dc5f8f9be5dc"
# We'll start by importing the required libraries, including the PEFT library and other dependencies.

# %% id="0a40abdf-ca1c-436f-a2af-603cd67a45a4"
import torch
import transformers
import peft
import tqdm
import numpy as np

# %% [markdown] id="a445f8be-545d-4085-a5f9-c64983655224"
# ### 2. Load models
#

# %% [markdown] id="63782b55-1c38-4e44-b003-e57daa813bed"
# We'll load a pre-trained DNA Language Model, "SpeciesLM", that serves as the base for fine-tuning. This is done using the transformers library from HuggingFace.
#
# The tokenizer and the model comes from the paper, "Species-aware DNA language models capture regulatory elements and their evolution". [Paper Link](https://www.biorxiv.org/content/10.1101/2023.01.26.525670v2), [Code Link](https://github.com/gagneurlab/SpeciesLM). They introduce a species-aware DNA language model, which is trained on more than 800 species spanning over 500 million years of evolution.

# %% id="dac961f4-c450-4124-923e-f4ba9bbd5e07"
from transformers import AutoTokenizer, AutoModelForMaskedLM

# %% id="e73fae58-03e9-4acc-b0fc-9bc810c7d366"
tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")
lm = AutoModelForMaskedLM.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")

# %% colab={"base_uri": "https://localhost:8080/"} id="ca43b893-2d66-4e93-a08f-b17a92040709" outputId="ccbac964-a329-414d-f537-3cae7da66cf2"
lm.eval()
lm.to("cuda");

# %% [markdown] id="c1bda6f2-34bb-4ce2-aa3f-3013548b0a28"
# ### 2. Prepare datasets

# %% [markdown] id="f4c61e59-457c-47d9-8929-5e8cd32d3125"
# We'll load the `nucleotide_transformer_downstream_tasks` dataset, which contains 18 downstream tasks from the Nucleotide Transformer paper. This dataset provides a consistent genomics benchmark with binary classification tasks.

# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["03bba232d3974119acf8031bc086a072", "9107f7bfc8d3483390f802b0458e9380", "f5c80fa70ead4c86aa3b2a046061b901", "57966a469ca1458daab74e81672ae855", "1464502dc3dd46308be8b4fcc9d5ddb9", "92f64c7e088342b9b3c070ba7a295ed0", "ab0aa8af3816422e9d97934f12af842c", "ff89a891bd9c42a8be164587a94ccac1", "e113a50f8ed2410ca12ce7cb38a1681d", "1afa6e9b69c74136863b7747e62a0608", "0838d19b226d486285a26ce0b04d7e15", "7bdab33f4b244fc89408b91755bf17c5", "4d4ce0d35c124690b3427e84a9a128b1", "33be6b0ca8fd44188f834a48a9574a72", "74e9bc1ead434ae78077df6b85f1df58", "e1acc6e70b9246a5b063b3e262f01c81", "078c6877377a491d97d6fadd27064a76", "d46ee1c39bac44c2b541a88c883de1cb", "12f1de7122a7471e90f01d9e7be81178", "dad286d42a514c9ca6bb01bfe9e9c4be", "c028ed977b5e479fbd93b8add588a6dc", "6d80dec073e449efba272fa9f3527922", "c311b777514f41ef986756a386c0bb34", "e2e4bf053ce442f6aee6ffab5f76525f", "c88cf701e20b4354a63ac7d8645d1df9", "f71c252ada474be882b0335ed9a0a1c3", "e059c665229e46ea905dcbd6fc179c88", "bd5273325a4b453e8053d98a09fe9493", "8f20ed2b74d84e80a8d403793354adea", "57c9af47364d48ffbb4ffbdd2c951ede", "fa9d75fcb1d5400c8ca1d1d13d28d0c7", "682644a713b145f0b2dcff99790c6d4d", "9b9b9d573d44464f9a6f5030a40245fe", "ec165fdbe87a4b00a6c288ef1e85c0a9", "17859b793a304e389d1ea0b9ccc3646f", "34921fd116cc42b7b530174d9f61e71e", "2d5466a5e98849c5a09f16faa98f91da", "952397f9c91c480184fa57e175ab1b4c", "86bcccb842244f4f9add58f62facaace", "78b5bbf4c8ac4fe5961776fded4d5798", "c80062a855cb41a28ac625ab03635da2", "aecd740c17c84d45b0615d4fc4196035", "39640709e7174f84a50da05764abbf99", "7114a029e75c4ed5b966eddd3a3c919d"]} id="f5c0b3df-911a-4645-9140-99ee489515e8" outputId="15315be1-9d07-4c46-acda-c65cb5a05250"
from datasets import load_dataset

raw_data = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", "H3")

# %% [markdown] id="bbb527c5-8077-4ce4-b093-ae627a5f253c"
# We'll use the "H3" subset of this dataset, which contains a total of 13,468 rows in the training data, and 1497 rows in the test data.

# %% colab={"base_uri": "https://localhost:8080/"} id="efef4bb2-60d8-40d1-8777-2b665a87059c" outputId="1c8526ce-0fcb-4fbc-d592-f9a6eae6ebdb"
raw_data

# %% [markdown] id="aafd37c8-6830-4070-a73b-cf62e72e901c"
# The dataset consists of three columns, ```sequence```, ```name``` and ```label```. An row in this dataset looks like:

# %% colab={"base_uri": "https://localhost:8080/"} id="eecd39d8-c073-4d3e-940e-fd83d46f83ab" outputId="0b5f8800-eb5d-4c41-a2bc-73e4f837a4d8"
raw_data['train'][0]

# %% [markdown] id="92eccf3e-e846-4c59-af56-0e336ac5a1cd"
# We split out dataset into training, test, and validation sets.

# %% id="f0649bbd-e74e-4dd6-a564-c4d65e46dbbf"
from datasets import Dataset, DatasetDict

train_valid_split = raw_data['train'].train_test_split(test_size=0.15, seed=42)

train_valid_split = DatasetDict({
    'train': train_valid_split['train'],
    'validation': train_valid_split['test']
})

ds = DatasetDict({
    'train': train_valid_split['train'],
    'validation': train_valid_split['validation'],
    'test': raw_data['test']
})


# %% [markdown] id="5424726f-a7ba-45d5-b449-36be9a98b8e6"
# Then, we use the tokenizer and a utility function we created, ```get_kmers``` to generate the final data and labels. The ```get_kmers``` function is essential for generating overlapping 6-mers needed by the language model (LM). By using k=6 and stride=1, we ensure that the model receives continuous and overlapping subsequences, capturing the local context within the biological sequence for more effective analysis and prediction.
#
#

# %% id="75f267a9-82d1-4343-982e-9b1ea542a330"
def get_kmers(seq, k=6, stride=1):
    return [seq[i:i + k] for i in range(0, len(seq), stride) if i + k <= len(seq)]


# %% id="efa9441d-f44c-4ca3-b24c-fa5c853896cd"
test_sequences = []
train_sequences = []
val_sequences = []

dataset_limit = 200 # NOTE: This dataset limit is set to 200, so that the training runs faster. It can be set to None to use the
                    # entire dataset

for i in range(0, len(ds['train'])):

    if dataset_limit and i == dataset_limit:
        break

    sequence = ds['train'][i]['sequence']
    sequence = "candida_glabrata " + " ".join(get_kmers(sequence))
    sequence = tokenizer(sequence)["input_ids"]
    train_sequences.append(sequence)


for i in range(0, len(ds['validation'])):
    if dataset_limit and i == dataset_limit:
        break
    sequence = ds['validation'][i]['sequence']
    sequence = "candida_glabrata " + " ".join(get_kmers(sequence))
    sequence = tokenizer(sequence)["input_ids"]
    val_sequences.append(sequence)


for i in range(0, len(ds['test'])):
    if dataset_limit and i == dataset_limit:
        break
    sequence = ds['test'][i]['sequence']
    sequence = "candida_glabrata " + " ".join(get_kmers(sequence))
    sequence = tokenizer(sequence)["input_ids"]
    test_sequences.append(sequence)


train_labels = ds['train']['label']
test_labels = ds['test']['label']
val_labels = ds['validation']['label']

if dataset_limit:
    train_labels = train_labels[0:dataset_limit]
    test_labels = test_labels[0:dataset_limit]
    val_labels = val_labels[0:dataset_limit]

# %% [markdown] id="0686955c-201a-427b-8bef-5c663edb85b8"
# Finally, we create a Dataset object for each our sets.

# %% id="445b4279-2446-46d6-af2a-ceb2638955c7"
from datasets import Dataset

train_dataset = Dataset.from_dict({"input_ids": train_sequences, "labels": train_labels})
val_dataset = Dataset.from_dict({"input_ids": val_sequences, "labels": val_labels})
test_dataset = Dataset.from_dict({"input_ids": test_sequences, "labels": test_labels})

# %% [markdown] id="d05d51a7-b933-4793-95df-af7d4d510b13"
# ### 4. Train model

# %% [markdown] id="b5ce1985-c24e-4feb-a6d4-aacb909536f0"
# Now, we'll train our DNA Language Model with the training dataset. We'll add a linear layer in the final layer of our language model, and then, train all the parameteres of our model with the training dataset.

# %% id="3a34b2c0-6205-4d48-b1a6-371b50ca42de"
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% colab={"base_uri": "https://localhost:8080/"} id="700540f4-0ab8-4f8a-a75c-416a6908af47" outputId="9e16c1e9-4676-4cdf-b2a9-d785773b1c8d"
import torch
from torch import nn

class DNA_LM(nn.Module):
    def __init__(self, model, num_labels):
        super(DNA_LM, self).__init__()
        self.model = model.bert
        self.in_features = model.config.hidden_size
        self.out_features = num_labels
        self.classifier = nn.Linear(self.in_features, self.out_features)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1]
        # Use the [CLS] token for classification
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits

# Number of classes for your classification task
num_labels = 2
classification_model = DNA_LM(lm, num_labels)
classification_model.to('cuda');

# %% id="0af97341-8f95-41d9-9d91-1eb64da4b516"
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% colab={"base_uri": "https://localhost:8080/", "height": 268} id="d9ce6bc3-4f63-4b7b-b28d-d2553002e6db" outputId="0c8fdbad-f34d-492b-e146-db6c2064e7c5"
from transformers import Trainer, TrainingArguments


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_steps=1,
    logging_steps=1,
)

# Initialize Trainer
trainer = Trainer(
    model=classification_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# %% [markdown] id="ebc7e33a-caad-4412-84e3-3e1ce7d02ccd"
# ### 5. Evaluation

# %% colab={"base_uri": "https://localhost:8080/", "height": 124} id="38eb0273-ce7e-4770-8457-2f9609f6843b" outputId="2b0b93c9-0199-4e71-9825-9f6a2bd199d0"
# Generate predictions

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
predicted_labels = logits.argmax(axis=-1)
print(predicted_labels)


# %% [markdown] id="ae4c7bca"
# Then, we create a function to calculate the accuracy from the test and predicted labels.

# %% colab={"base_uri": "https://localhost:8080/"} id="327a1c3b-88d6-4430-8978-73a7cbdbb697" outputId="f03ad54d-d35f-4fcc-e709-c24d14906e25"
def calculate_accuracy(true_labels, predicted_labels):

    assert len(true_labels) == len(predicted_labels), "Arrays must have the same length"
    correct_predictions = np.sum(true_labels == predicted_labels)
    accuracy = correct_predictions / len(true_labels)

    return accuracy

accuracy = calculate_accuracy(test_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# %% [markdown] id="9p0fFXKTZz9Q"
# The results aren't that good, which we can attribute to the small dataset size.

# %% [markdown] id="e681864c-f15a-40a6-ac34-0e631d68d5c8"
# ### 7. Parameter Efficient Fine-Tuning Techniques

# %% [markdown] id="9141fabe-417b-4fbb-bd3e-244ad84e3010"
# In this section, we demonstrate how to employ parameter-efficient fine-tuning (PEFT) techniques to adapt a pre-trained model for specific genomics tasks using the PEFT library.

# %% [markdown] id="71b8a749-461e-4533-b1d0-cebc924d3dc0"
# The LoraConfig object is instantiated to configure the PEFT parameters:
#
# - task_type: Specifies the type of task, in this case, sequence classification (SEQ_CLS).
# - r: The rank of the LoRA matrices.
# - lora_alpha: Scaling factor for adaptive re-parameterization.
# - target_modules: Modules within the model to apply PEFT re-parameterization (query, key, value in this example).
# - lora_dropout: Dropout rate used during PEFT fine-tuning.

# %% colab={"base_uri": "https://localhost:8080/"} id="021641ae-f604-4d69-8724-743b7d7c613c" outputId="d7c41fca-1c6b-46fd-9116-01f42d1d6ddf"
# Number of classes for your classification task
num_labels = 2
classification_model = DNA_LM(lm, num_labels)
classification_model.to('cuda');

# %% id="6c223937-86ea-42ef-991a-050f23b21ef9"
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.01,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="e7a9fe7d-e3ac-4ffa-9a9b-2067fb09b885" outputId="02a6c65f-7474-4bc1-bfab-c05532e350a5"
from peft import get_peft_model

peft_model = get_peft_model(classification_model, peft_config)
peft_model.print_trainable_parameters()

# %% colab={"base_uri": "https://localhost:8080/"} id="22064519-eaab-4142-8618-d1210d05c6bd" outputId="ca3f764d-cdb4-4525-c541-8eabfb4cde57"
peft_model

# %% colab={"base_uri": "https://localhost:8080/", "height": 268} id="d3812e96-6b49-4911-8b21-d8871b7c06a5" outputId="8d497e30-1d3f-457a-f62a-244731698cb2"
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_steps=1,
    logging_steps=1,
)

# Initialize Trainer
trainer = Trainer(
    model=peft_model.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# %% [markdown] id="76dbd948-d919-4ade-a405-cec297979577"
# ### 8. Evaluate PEFT Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 124} id="58cf70ba-47d5-4111-bb12-830ae04c6285" outputId="0abc56a9-bd68-4e4e-9f13-756e8c9ffa3e"
# Generate predictions

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
predicted_labels = logits.argmax(axis=-1)
print(predicted_labels)


# %% colab={"base_uri": "https://localhost:8080/"} id="4bd38fe5-6513-4c88-afee-0cc4e1781fdd" outputId="a50a91d0-d04d-4620-9006-868716bb992d"
def calculate_accuracy(true_labels, predicted_labels):

    assert len(true_labels) == len(predicted_labels), "Arrays must have the same length"
    correct_predictions = np.sum(true_labels == predicted_labels)
    accuracy = correct_predictions / len(true_labels)

    return accuracy

accuracy = calculate_accuracy(test_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# %% [markdown]
# As we can see, the PEFT model achieves similar performance to the baseline model, demonstrating the effectiveness of PEFT in adapting pre-trained models to specific tasks with limited computational resources.
#
# With PEFT, we only train 442,368 parameters, which is 0.49% of the total parameters in the model. This is a significant reduction in computational resources compared to training the entire model from scratch.
#
# We can improve the results by using a larger dataset, fine-tuning the model for more epochs or changing the hyperparameters (rank, learning rate, etc.).
#
