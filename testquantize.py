from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

#file testing different quantized Mixtral models
#ISSUE: the 7B models work fine,  other ones give a CUDA error...

#model_name = "jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit"
#model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
model_name = "TheBloke/Instruct_Mixtral-8x7B-v0.1_Dolly15K-GPTQ"
#model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"




tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

text = "Hello my name is"
messages = [
    # {"role": "user", "content": "What is your favourite condiment?"},
    # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Can everything be proven or disproven? Explain why or why not."}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#model.save_pretrained("testfolder", from_pt=True) 