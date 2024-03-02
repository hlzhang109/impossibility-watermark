import re
import pandas as pd 
import os
from openai import OpenAI
import random
import hydra

# Set your OpenAI API key here
from dotenv import load_dotenv, find_dotenv

from pipeline_builder import PipeLineBuilder


load_dotenv(find_dotenv()) # load openai api key from ./.env

def find_csv(txt_file_path):
    with open(txt_file_path, 'r') as file:
        content = file.read()

    # Search for the first occurrence of 'attack_*.csv' pattern
    match = re.search(r'attack_.*\.csv', content)

    if match:
        csv_filename = match.group(0)
        return csv_filename
    return None

def get_mutated_text(txt_file_path):
    file_name = find_csv(txt_file_path)
    directory = "./eval/results/"
    file_path = os.path.join(directory, file_name)

    df = pd.read_csv(file_path)
    
    success_df = df[(df['mutated_text_len'] >= 0.95 * df['current_text_len']) & (df['quality_preserved'] == True)]

    return success_df['mutated_text'].iloc[-1]


def query_gpt4_with_history(initial_prompt, follow_up_prompt):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt}
    ]
    )

    first_response = completion.choices[0].message
    
    completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt},
        # {'role': "assistant", "content": first_response},
        {"role": "user", "content": follow_up_prompt},
    ]
    )

    second_response = completion.choices[0].message
    
    return first_response, second_response

def distinguish():
    
    prompt = """Write a 250-word story about Evan, a married American tourist, who falls for Emilie, a lively barista, during a spring festival in Paris. They bond over their love for art and the city's charm, leading to walks along the Seine, where the festival's joy spills over. Their connection deepens amid shared laughter and explorations of quaint streets. As the festival lights dance on the river, Evan shares his feelings with Emilie under the starlit sky, promising to cherish the moments they've shared. Focus on their joyful adventures and the sincere emotions that bloom between them."""
    response_2 = """Evan, an American tourist, found himself captivated not just by the allure of Paris in spring but by Emilie, a lively barista with a smile that mirrored the city's charm. Their connection sparked during a bustling spring festival, where art and joy intertwined, drawing them closer over shared passions.\n\nTheir days were filled with wanderings along the Seine, where the festival's exuberance seemed to overflow, blending seamlessly with the serene flow of the river. Laughter became their shared language as they explored the city's quaint streets and hidden gems, with Emilie introducing Evan to the true essence of Paris beyond the postcards.\n\nArt, a mutual love, allowed them to see the world through each other's eyes. They lost themselves in galleries and street art, finding pieces of themselves in the colors and shapes. The city, with its endless charm, became the backdrop for their growing connection, a testament to the unexpected paths of the heart.\n\nAs the festival lights danced on the river's surface, Evan found the courage under the starlit sky to share his feelings with Emilie. Amid the magic of the moment, he spoke of the joy she had brought into his life, promising to cherish the memories they had created together. Their story, though fleeting, was a vivid tapestry of laughter, exploration, and sincere emotions, a reminder of the beauty found in shared experiences and the unpredictable journey of the heart."""
    response_1 = """Evan, an American tourist, found himself captivated by the vibrant spirit of Paris during a spring festival. Amidst the lively streets adorned with blossoms and the air filled with joyous melodies, he stumbled upon a quaint café where Emilie, a barista with an infectious smile, brewed magic in a cup. Their initial exchange over a love for art and the city's undying charm quickly blossomed into a series of adventures along the Seine, where the festival's exuberance seemed to mirror their growing affection.
Their walks were filled with shared laughter and endless conversations, as they explored the city's quaint streets and hidden gems. Paris, in its festival attire, provided the perfect backdrop for their budding connection, with its cobbled lanes and the Seine's banks lit by the festival lights, dancing on the water like fireflies under the starlit sky.
One evening, as they found themselves under the canopy of stars, with the city's lights reflecting in their eyes, Evan shared his feelings with Emilie. The joy and sincerity in his voice resonated with the genuine emotions that had taken root in their hearts. He promised to cherish the moments they had shared, acknowledging the special bond they had formed. In the heart of Paris, amidst the spring festival's enchantment, Evan and Emilie discovered a connection that transcended the ordinary, marked by sincere emotions and joyful adventures, forever etched in their memories."""
    initial_prompt = f"""Here's a prompt and two possible responses. Compare the responses and explain the difference between them.

Prompt

{prompt}

Response 1

{response_1}

Response 2

{response_2}
    """
    
    txt_file_directory = "./third_round/"
    txt_file_name = "3_evan_2_1.txt"
    txt_file_path = os.path.join(txt_file_directory, txt_file_name)
    
    perturbed_1 = get_mutated_text(txt_file_path)
    
    txt_file_directory = "./third_round/"
    txt_file_name = "3_evan_3_2.txt"
    txt_file_path = os.path.join(txt_file_directory, txt_file_name)
    
    perturbed_2 = get_mutated_text(txt_file_path)
    
    if random.choice([True, False]):
        perturbed_1, perturbed_2 = perturbed_2, perturbed_1
        print("Flipped.")
    
    
    follow_up_prompt = f"""Here are two completely different responses. See if you can match these responses to Response 1 and Response 2. At the end of your response, output 1 if you match Response A with Response 1 and 2 if you match response A with response 2.

Response A
{perturbed_1}

Response B

{perturbed_2}"""
    
    first_response, second_response = query_gpt4_with_history(initial_prompt, follow_up_prompt)

    print("First Response:", first_response)
    print("Second Response:", second_response)
    
if __name__ == "__main__":
    distinguish()