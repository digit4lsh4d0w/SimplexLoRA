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

# %% id="jONLwzXgLg-I"
# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q git+https://github.com/huggingface/peft.git
# !pip install -q git+https://github.com/huggingface/accelerate.git@main
# !pip install huggingface_hub
# !pip install bitsandbytes
# !pip install SentencePiece

# %% id="36460935"
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %% id="1351e04c"
from huggingface_hub import notebook_login
import torch

notebook_login()

# %% id="d85af699"
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsAndBytesConfig

model_name = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto", use_auth_token=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="f0f515ed" outputId="312488a5-f4f8-48a4-8c63-7b4a59e80418"
# %%time
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b", adapter_name="eng_alpaca")

# %% id="67a0c121"
# %%time
model.load_adapter("22h/cabrita-lora-v0-1", adapter_name="portuguese_alpaca")

# %% id="4b655fca"
model

# %% id="e9ebd572"
model.to("cuda")

# %% id="138805b3"
import torch

device = "cuda"


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


# %% colab={"base_uri": "https://localhost:8080/"} id="fd5e6b3b" outputId="ec72241b-c427-4258-b02f-2101df0d171a"
# %%time
model.set_adapter("eng_alpaca")

# %% colab={"base_uri": "https://localhost:8080/"} id="33650851" outputId="aae24052-0f09-4812-88c3-6fb53dec656c"
instruction = "Tell me about alpacas."

print(evaluate(instruction))

# %% colab={"base_uri": "https://localhost:8080/"} id="fdc7196e" outputId="44cb6742-066b-470e-f507-cbf21e5ae030"
# %%time
model.set_adapter("portuguese_alpaca")

# %% colab={"base_uri": "https://localhost:8080/"} id="31997da3" outputId="8071de75-dc9d-4e89-e85f-674f1de22658"
instruction = "Invente uma desculpa criativa pra dizer que não preciso ir à festa."

print(evaluate(instruction))

# %% colab={"base_uri": "https://localhost:8080/"} id="8b8e4e9a" outputId="84226223-e018-4feb-e189-969c344fd940"
with model.disable_adapter():
    instruction = "Invente uma desculpa criativa pra dizer que não preciso ir à festa."

    print(evaluate(instruction))
