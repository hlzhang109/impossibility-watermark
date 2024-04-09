import argparse
import os
import random
import logging
from utils import get_mutated_text, query_openai_with_history, get_prompt_or_output, get_watermarked_text, get_nth_successful_perturbation
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# def generate_comparison_prompts(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose):
#     prompt_1 = f"""Given a prompt and two responses, explain how the responses differ.
# Prompt: {prompt}
# Response 1: {response_1}
# Response 2: {response_2}"""
#     if verbose:
#         prompt_2 = f"""Given two new responses to the same prompt, match the new responses to the old responses. Output 1 if Response A matches with Response 1 and Response B matches with Response 2, and output 2 if Response B matches with Response 1 and Response A matches with response 2. Explain your reasoning.
# Response A: {perturbed_1}
# Response B: {perturbed_2}"""
#     else:
#         prompt_2 = f"""Given two new responses to the same prompt, match the new responses to the old responses. Output 1 if Response A matches with Response 1 and Response B matches with Response 2, and output 2 if Response B matches with Response 1 and Response A matches with response 2. Respond with the output only.
# Response A: {perturbed_1}
# Response B: {perturbed_2}"""
    
#     return prompt_1, prompt_2

def generate_comparison_prompts(prompt, response_1, response_2, perturbed):
    prompt_1 = f"""Given a prompt and two responses, explain how the responses differ.
Prompt: {prompt}
Response 1: {response_1}
Response 2: {response_2}"""

    prompt_2 = f"""Here's a third response. Output 1 if it matches to the first prompt and 2 otherwise. Explain your reasoning.

{perturbed}"""

    return prompt_1, prompt_2

def generate_comparison_prompts_v2(prompt, response_1, response_2, perturbed):
    prompt_1 = f"""Given a prompt and three responses, explain how the responses differ.
Prompt: {prompt}
Response 1: {perturbed}
Response 2: {response_1}
Response 3: {response_2}"""

    prompt_2 = f"""Now, match response 1 to response 2 or 3. If you match to 2, respond with a 2. Otherwise, respond with a 3. Afterwards, explain your reasoning."""

    return prompt_1, prompt_2

def distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=False):
    flipped = False
    if random.choice([True, False]):
        flipped = True
        perturbed = perturbed_2
        # perturbed_1, perturbed_2 = perturbed_2, perturbed_1
        logging.info("Flipped.")
    else:
        perturbed = perturbed_1
    
    # prompt_1, prompt_2 = generate_comparison_prompts(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose)
    prompt_1, prompt_2 = generate_comparison_prompts(prompt, response_1, response_2, perturbed)
    
    first, second = query_openai_with_history(prompt_1, prompt_2)
    logging.info(f"Model's First Response: {first.content}")
    logging.info(f"Model's Second Response: {second.content}")
    logging.info("---------------------------------------------------------------")

    good_trial = second.content[0] in "12"
    match_success = good_trial and (not flipped and second.content[0] == "1" or flipped and second.content[0] == "2")
    return flipped, good_trial, match_success

def distinguish_v2(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=False):
    response_flipped = False
    if random.choice([True, False]):
        response_flipped = True
        response_2, response_1 = response_1, response_2
        logging.info("Responses flipped.") 
    
    flipped = False
    if random.choice([True, False]):
        logging.info("Answers flipped.") 
        flipped = True
        perturbed = perturbed_2
    else:
        perturbed = perturbed_1
        
    prompt_1, prompt_2 = generate_comparison_prompts_v2(prompt, response_1, response_2, perturbed)
    
    first, second = query_openai_with_history(prompt_1, prompt_2)
    logging.info(f"Model's First Response: {first.content}")
    logging.info(f"Model's Second Response: {second.content}")
    logging.info("---------------------------------------------------------------")
    
    overall_flipped = response_flipped ^ flipped

    good_trial = second.content[0] in "23"
    match_success = good_trial and (not overall_flipped and second.content[0] == "2" or overall_flipped and second.content[0] == "3")
    return overall_flipped, good_trial, match_success    

def main():
    parser = argparse.ArgumentParser(description="Distinguish perturbed responses.")
    parser.add_argument("entropy", type=int, help="Entropy value")
    parser.add_argument("output_1", type=int, help="First output number")
    parser.add_argument("attack_id_1", type=str, help="First attack ID")
    parser.add_argument("output_2", type=int, help="Second output number")
    parser.add_argument("attack_id_2", type=str, help="Second attack ID")
    parser.add_argument("--log_suffix", type=str, required=False, default="",help="Log suffix")
    parser.add_argument("--num_trials", type=int, required=False, default=10, help="Number of trials")
    parser.add_argument("--mutation_num", type=int, required=False, default = -1, help="The nth successful mutation.")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")

    args = parser.parse_args()
    
    entropy = args.entropy
    output_1 = args.output_1
    attack_id_1 = args.attack_id_1
    output_2 = args.output_2
    attack_id_2 = args.attack_id_2
    log_suffix = args.log_suffix
    num_trials = args.num_trials
    verbose = args.verbose
    mutation_num = args.mutation_num
    
    # Construct log filename based on command line arguments
    log_filename = f"./results/stationary_distribution/robustness_analysis/entropy_{entropy}/distinguisher_results/{output_1}_{attack_id_1}-{output_2}_{attack_id_2}{log_suffix}.log"
    
    # Configure logging to dynamically created filename
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')
    
    prompts_file_path = './inputs/dynamic_prompts.csv'
    prompt = get_prompt_or_output(prompts_file_path, entropy) 
    
    # Get Perturbed Versions
    csv_file_directory = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/"
    
    first_perturbed_csv_filename = f"output_{output_1}/corpuses/attack_{attack_id_1}.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    response_1 = get_watermarked_text(csv_file_path)
    perturbed_1 = get_nth_successful_perturbation(csv_file_path, mutation_num)
    
    second_perturbed_csv_filename = f"output_{output_2}/corpuses/attack_{attack_id_2}.csv"
    csv_file_path = os.path.join(csv_file_directory, second_perturbed_csv_filename)
    response_2 = get_watermarked_text(csv_file_path)
    perturbed_2 = get_nth_successful_perturbation(csv_file_path, mutation_num)

    
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Response 1: {response_1}")
    logging.info(f"Response 2: {response_2}")
    logging.info(f"Perturbed 1: {first_perturbed_csv_filename}")
    logging.info(f"Perturbed 2: {second_perturbed_csv_filename}")

    num_regular_trials = 0
    num_flipped_trials = 0
    num_good_trials = 0
    num_regular_good_trials = 0
    num_flipped_good_trials = 0
    num_regular_success_matches = 0
    num_flipped_success_matches = 0
    num_success_matches = 0
    
    for _ in range(num_trials):
        flipped, good_trial, match_success = distinguish_v2(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=verbose)
        
        # Increment trials counters
        if flipped:
            num_flipped_trials += 1
        else:
            num_regular_trials += 1

        # Increment counters for good trials and success matches
        if good_trial:
            num_good_trials += 1
            if flipped:
                num_flipped_good_trials += 1
            else:
                num_regular_good_trials += 1
            
            if match_success:
                num_success_matches += 1
                if flipped:
                    num_flipped_success_matches += 1
                else:
                    num_regular_success_matches += 1
     
    regular_distinguish_accuracy = float(num_regular_success_matches)/num_regular_good_trials
    flipped_distinguish_accuracy = float(num_flipped_success_matches)/num_flipped_good_trials
    distinguish_accuracy = float(num_success_matches)/num_good_trials
    
    logging.info(f"Number of attempted trials: {num_trials}")
    logging.info(f"Number of good trials: {num_good_trials}")
    logging.info(f"Number of correct matches: {num_success_matches}")
    logging.info(f"Distinguisher accuracy: {distinguish_accuracy}")
    logging.info(f"-" * 50)
    logging.info(f"Number of regular attempted trials: {num_regular_trials}")
    logging.info(f"Number of regular good trials: {num_regular_good_trials}")
    logging.info(f"Number of regular correct matches: {num_regular_success_matches}")
    logging.info(f"Regular distinguisher accuracy: {regular_distinguish_accuracy}")
    logging.info(f"-" * 50)
    logging.info(f"Number of flipped attempted trials: {num_flipped_trials}")
    logging.info(f"Number of flipped good trials: {num_flipped_good_trials}")
    logging.info(f"Number of flipped correct matches: {num_flipped_success_matches}")
    logging.info(f"Flipped distinguisher accuracy: {flipped_distinguish_accuracy}")
    logging.info(f"-" * 50)    

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()