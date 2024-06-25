import ast
from functools import partial
from datasets import load_dataset, concatenate_datasets, Dataset
import time

from benchmark.annotators import *

# Step 1: Download Dataset
dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train", cache_dir="/data2/.shared_datasets/")
print(f"Original Dataset: {dataset}")

# Step 2: Clean Dataset
def convert_columns_to_utf8(dataset: Dataset) -> Dataset:
    def convert_to_utf8(value):
        if isinstance(value, str):
            return value.encode('utf-8').decode('utf-8')
        return value

    for column in dataset.column_names:
        dataset = dataset.map(lambda x: {column: convert_to_utf8(x[column])}, batched=False)

    return dataset

dataset = convert_columns_to_utf8(dataset)

# Step 3: Filter Dataset

# 3.1. Filter dataset for length 
output_low, output_high = 100, 1000
dataset = dataset.filter(lambda example: output_low <= len(example["response_a"].split()) <= output_high)
dataset = dataset.filter(lambda example: output_low <= len(example["response_b"].split()) <= output_high)

print(f"Dataset after length filtering: {dataset}")

# 3.2. Filter dataset for amount of response turns
num_turns = 1
dataset = dataset.filter(lambda example: len(ast.literal_eval(example["prompt"])) == num_turns)

print(f"Dataset after response turn filtering: {dataset}")

# 3.3. Control number of winner types
winner_a_amount, winner_b_amount, winner_tie_amount = 5, 5, 5
winner_model_a = dataset.filter(lambda x: x['winner_model_a'] == 1).select(range(winner_a_amount))
winner_model_b = dataset.filter(lambda x: x['winner_model_b'] == 1).select(range(winner_b_amount))
winner_tie     = dataset.filter(lambda x: x['winner_tie']     == 1).select(range(winner_tie_amount))
dataset        = concatenate_datasets([winner_model_a,winner_model_b,winner_tie]).shuffle()

print(f"Dataset after winner type filtering: {dataset}")

# Step 4: Reformat Dataset
def convert_list_to_str(example, key="prompt"):
    try:
        return {key: ast.literal_eval(example[key])[0]}
    except:
        return None

dataset = dataset.map(partial(convert_list_to_str, key="prompt"))
dataset = dataset.map(partial(convert_list_to_str, key="response_a"))
dataset = dataset.map(partial(convert_list_to_str, key="response_b"))

# Step 5: Apply Annotators

def apply_annotation(example, llm, annotation_fn, input_keys, output_keys, persona=None):
    inputs = {k: example[k] for k in input_keys}
    output = llm + annotation_fn(**inputs, persona=persona)
    return {k: output[k] for k in output_keys}

model_id = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"

# Load the model
llm = models.Transformers(
    model_id, 
    echo=False,
    cache_dir="/data2/.shared_models/", 
    device_map='auto'
)

entroy_persona = \
"""
You are an Entropy Analyst, specializing in evaluating the unpredictability, variety, and informativeness of texts. 
Your expertise in linguistic analysis allows you to assess how diverse and unexpected the content is based on given instructions. 
You approach each task with objectivity and a keen attention to detail, ensuring your evaluations are impartial and thorough. 
Your goal is to provide clear, concise, and accurate assessments of the responses.
"""

# input_keys = ["prompt"]
# output_keys = ["entropy_level_prompt"]
# dataset = dataset.map(
#     partial(apply_annotation, 
#             llm=llm, 
#             annotation_fn=annotate_entropy_by_instructions, 
#             input_keys=input_keys, 
#             output_keys=output_keys, 
#             persona=entroy_persona,
#     )
# )

# print(f"Dataset after adding entropy annotation: {dataset}")

input_keys = ["prompt"]
output_keys = ["entropy_level_prompt_w_exp"]
dataset = dataset.map(
    partial(apply_annotation, 
            llm=llm, 
            annotation_fn=annotate_entropy_by_instructions_w_exp, 
            input_keys=input_keys, 
            output_keys=output_keys, 
            persona=entroy_persona,
    )
)

print(f"Dataset after adding entropy annotation: {dataset}")

# input_keys = ["prompt", "response_a", "response_b"]
# output_keys = ["entropy_level_prompt_and_responses"]
# dataset = dataset.map(
#     partial(apply_annotation, 
#             llm=llm, 
#             annotation_fn=annotate_entropy_by_instructions_and_responses, 
#             input_keys=input_keys, 
#             output_keys=output_keys, 
#             persona=entroy_persona,
#     )
# )

# print(f"Dataset after adding entropy annotation: {dataset}")

df = dataset.to_pandas()
df.to_csv("./benchmark/sample.csv")

# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0 python -m benchmark.create