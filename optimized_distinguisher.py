import argparse
import os
import random
import logging
from utils import get_mutated_text, query_openai_with_history, get_prompt_or_output, get_watermarked_text, get_nth_successful_perturbation
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def generate_comparison_prompts(prompt, response_1, response_2, perturbed):
    prompt_1 = f"""
Story 1: {response_1}
Story 2: {response_2}
Story A: {perturbed}

Story A was created by modifying Story 1 or Story 2. Which one? Explain your reasoning in-depth before responding. Both stories have similar themes and plots, so focus on specific details to
make a decision."""

    prompt_2 = f"""So, what's your decision? Was Story A created by modifying Story 1 or Story 2? Respond with 1 if it was created by modifying Story 1, and 2 if it was created by modifying Story 2."""

    return prompt_1, prompt_2

def distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=False):
    regular_flipped = False
    if random.choice([True, False]):
        regular_flipped = True
        response_1, response_2 = response_2, response_1
        logging.info("Stories flipped.")
    
    perturbed_flipped = False
    if random.choice([True, False]):
        perturbed_flipped = True
        perturbed = perturbed_2
        logging.info("Perturbations flipped.")
    else:
        perturbed = perturbed_1
    flipped = regular_flipped ^ perturbed_flipped
    if flipped:
        logging.info("Overall flipped.")
        
    prompt_1, prompt_2 = generate_comparison_prompts(prompt, response_1, response_2, perturbed)
        
    for _ in range(2):
        first, second = query_openai_with_history(prompt_1, prompt_2)
        logging.info(f"Model's First Response: {first.content}")
        logging.info(f"Model's Second Response: {second.content}")
        logging.info("---------------------------------------------------------------")
        
        decision = second.content[0]
        good_trial = decision in "12"
        if good_trial == 1:
            break
        
    match_success = good_trial and (not flipped and second.content[0] == "1" or flipped and second.content[0] == "2")
    
    return regular_flipped, perturbed_flipped, good_trial, match_success

def calculate_condition_success_rate(results, regular_flipped_condition, perturbed_flipped_condition):
    """
    Calculates the success rate for trials matching the given flip conditions.

    Parameters:
    - results: List of tuples with trial results in the form (regular_flipped, perturbed_flipped, good_trial, match_success).
    - regular_flipped_condition: The desired condition for regular_flipped (True or False).
    - perturbed_flipped_condition: The desired condition for perturbed_flipped (True or False).

    Returns:
    - The success rate for the matching trials as a percentage.
    """
    total_trials = 0
    good_trials = 0
    successful_trials = 0

    for regular_flipped, perturbed_flipped, good_trial, match_success in results:
        # Check if the trial matches the specified conditions
        if regular_flipped == regular_flipped_condition and perturbed_flipped == perturbed_flipped_condition:
            total_trials += 1
            if good_trial:
                good_trials += 1
            if match_success:
                successful_trials += 1
                
    return successful_trials, good_trials, total_trials

def log_trial_results(results):
    """
    Logs the trial results for each combination of regular and perturbed conditions.

    Parameters:
    - results: List of tuples with trial results.
    """
    # Iterate over all combinations of conditions
    for regular in [True, False]:
        for perturbed in [True, False]:
            # Calculate success rates for the current combination
            successful_trials, good_trials, total_trials = calculate_condition_success_rate(results, regular, perturbed)

            success_rate = (successful_trials / total_trials) * 100 if total_trials > 0 else 0

            # Logging results for the current combination
            logging.info(f"Conditions - Regular Flipped: {regular}, Perturbed Flipped: {perturbed}")
            logging.info(f"Number of attempted trials: {total_trials}")
            logging.info(f"Number of good trials: {good_trials}")
            logging.info(f"Number of correct matches: {successful_trials}")
            logging.info(f"Distinguisher accuracy: {success_rate:.2f}%")
            logging.info(f"{'-' * 50}")

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
    
    results = []
    
    num_good_trials = 0
    num_success_matches = 0
    
    for _ in range(num_trials):
        curr_result = distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2, verbose=verbose)
        results.append(curr_result)
     
        if curr_result[2]:
            num_good_trials +=1
        if curr_result[3]:
            num_success_matches += 1
    
    distinguish_accuracy = float(num_success_matches) / num_good_trials
    
    logging.info(f"Number of regular attempted trials: {num_trials}")
    logging.info(f"Number of regular good trials: {num_good_trials}")
    logging.info(f"Number of regular correct matches: {num_success_matches}")
    logging.info(f"Regular distinguisher accuracy: {distinguish_accuracy}")
    logging.info(f"-" * 50)
    
    log_trial_results(results)
    
    
if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()