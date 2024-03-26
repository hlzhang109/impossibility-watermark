import os
import random
from utils import get_mutated_text, query_openai_with_history, get_prompt_or_output, get_watermarked_text
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def generate_comparison_prompts(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose):
    prompt_1 = f"""Given a prompt and two responses, give as concise an explanation as possible about their differences.
Prompt: {prompt}
Response 1: {response_1}
Response 2: {response_2}"""
    if verbose:
        prompt_2 = f"""Given two new responses to the same prompt, match the new responses to the old responses. Output 1 if New Response 1 matches with Response 1 and New Response 2 matches with Response 2, and output 2 if New Response 2 matches with Response 1 and New Response 1 matches with response 2. Give a short explanation for your reasoning.
New Response 1: {perturbed_1}
New Response 2: {perturbed_2}"""
    else:
        prompt_2 = f"""Given two new responses to the same prompt, match the new responses to the old responses. Output 1 if New Response 1 matches with Response 1 and New Response 2 matches with Response 2, and output 2 if New Response 2 matches with Response 1 and New Response 1 matches with response 2. Respond with the output only.
New Response 1: {perturbed_1}
New Response 2: {perturbed_2}"""
    
    return prompt_1, prompt_2

def distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=False):
    flipped = False
    if random.choice([True, False]):
        flipped = True
        perturbed_1, perturbed_2 = perturbed_2, perturbed_1
        print("Flipped.")
    
    prompt_1, prompt_2 = generate_comparison_prompts(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose)
    
    first, second = query_openai_with_history(prompt_1, prompt_2)
    print("Model's First Response:", first.content)
    print("Model's Second Response:", second.content)
    print("---------------------------------------------------------------")

    good_trial = second.content[0] in "12"
    match_success = good_trial and (not flipped and second.content[0] == "1" or flipped and second.content[0] == "2")
    return good_trial, match_success

def main():
    entropy = 5
    output_1 = 1
    attack_id_1 = "1_1"
    output_2 = 2
    attack_id_2 = "1_1"
    
    prompts_file_path = './inputs/dynamic_prompts.csv'
    prompt = get_prompt_or_output(prompts_file_path, 5) 
    
    # Get Perturbed Versions
    csv_file_directory = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/"
    
    first_perturbed_csv_filename = f"output_{output_1}/corpuses/attack_{attack_id_1}.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    response_1 = get_watermarked_text(csv_file_path)
    perturbed_1 = get_mutated_text(csv_file_path)
    
    second_perturbed_csv_filename = f"output_{output_2}/corpuses/attack_{attack_id_2}.csv"
    csv_file_path = os.path.join(csv_file_directory, second_perturbed_csv_filename)
    response_2 = get_watermarked_text(csv_file_path)
    perturbed_2 = get_mutated_text(csv_file_path)
    
    print(f"Prompt: {prompt}")
    print(f"Response 1: {response_1}")
    print(f"Response 2: {response_2}")
    print(f"Perturbed 1: {first_perturbed_csv_filename}")
    print(f"Perturbed 2: {second_perturbed_csv_filename}")

    verbose = False
    num_trials = 5
    
    num_good_trials = 0
    num_success_matches = 0
    
    for _ in range(num_trials):
        good_trial, match_success = distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=verbose)
        if good_trial:
            num_good_trials += 1
            if match_success:
                num_success_matches += 1
    
    distinguish_accuracy = num_success_matches/num_good_trials

    print("Number of attempted trials:", num_trials)
    print("Number of good trials:", num_good_trials)
    print("Number of correct matches:", num_success_matches)
    print("Distinguisher accuracy:", distinguish_accuracy)


if __name__ == "__main__":
    main()