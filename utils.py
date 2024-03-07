import re
import pandas as pd
import os
import json
import datetime
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def save_to_csv(data, filename, rewrite=False):
    df_out = pd.DataFrame(data)
    if os.path.exists(filename) and not rewrite:
        df_out.to_csv(filename, mode='a', header=False, index=False)  # Append without writing headers
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_out.to_csv(filename, index=False)  # Create new file with headers
    print(f"Data appended to {filename}")
    
def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
    
# Read the content of the text file
def find_csv(txt_file_path):
    with open(txt_file_path, 'r') as file:
        content = file.read()

    # Search for the first occurrence of 'attack_*.csv' pattern
    match = re.search(r'attack_.*\.csv', content)

    if match:
        csv_filename = match.group(0)
        return csv_filename
    return None

def count_words(text):
    if text is None:
        return 0
    return len(text.split())

def get_prompt_or_output(csv_path, num):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # Get the specific text based on num
    if num <= len(df) and num > 0:
        story = df.iloc[num - 1]['text']
    else:
        raise Exception(f"Index out of range.")
    
    return story

def get_mutated_text(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    success_df = df[(df['mutated_text_len'] >= 0.95 * df['current_text_len']) & (df['quality_preserved'] == True)]

    return success_df['mutated_text'].iloc[-1]

def get_last_step_num(csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Access the last 'step_num' value
    last_step_num = df['step_num'].iloc[-1]
    
    return last_step_num
    
def get_prompt_and_completion_from_json(file_path, index):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize lists to store prefixes and completions
    prefixes = []
    completions = []

    # Iterate through each element in the list
    for item in data:
        prefixes.append(item['Prefix'])
        completions.append(item['Completion'])
        
    prompt = prefixes[index]
    watermarked_text = completions[index] 
    
    return prompt, watermarked_text

def query_openai(prompt, model = "gpt-4-0125-preview", max_tokens = None):
    client = OpenAI()

    completion = client.chat.completions.create(
    model=model,
    max_tokens=max_tokens,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    )
    
    response = completion.choices[0].message
    
    return response.content

def get_completion_from_openai(prefix, max_tokens = None):
    completion = query_openai(prefix, max_tokens=max_tokens)
    completion = prefix + " " + completion
    return completion

def query_openai_with_history(initial_prompt, follow_up_prompt, model = "gpt-4-0125-preview"):
    client = OpenAI()

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt}
    ]
    )

    first_response = completion.choices[0].message
    
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": initial_prompt},
        # {'role': "assistant", "content": first_response},
        {"role": "user", "content": follow_up_prompt},
    ]
    )

    second_response = completion.choices[0].message
    
    return first_response, second_response

def get_perturbation_stats(step_num, current_text, mutated_text, quality_preserved, quality_analysis, watermark_detected, watermark_score, backtrack):
    perturbation_stats = [{
        "step_num": step_num,
        "current_text": current_text,
        "mutated_text": mutated_text, 
        "current_text_len": count_words(current_text),
        "mutated_text_len": count_words(mutated_text), 
        "quality_preserved": quality_preserved,
        "quality_analysis" : quality_analysis,
        "watermark_detected": watermark_detected,
        "watermark_score": watermark_score,
        "backtrack" : backtrack,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }]
    
    return perturbation_stats