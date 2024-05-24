import argparse
from collections import Counter
import os
import logging
import textwrap
from functools import partial
from utils import query_openai_with_history, get_prompt_or_output, get_watermarked_text, get_nth_successful_perturbation
from dotenv import load_dotenv, find_dotenv
from local_matcher import Distinguisher
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

load_dotenv(find_dotenv())

def generate_comparison_prompts(response_1, response_2, perturbed):
    prompt_1 = textwrap.dedent(f"""
Story 1: {response_1}
Story 2: {response_2}~
Story A: {perturbed}

Story A was created by modifying Story 1 or Story 2. Which one? Explain your reasoning in-depth before responding. Both stories have similar themes and plots, so focus on specific details to
make a decision.""")

    prompt_2 = f"""So, what's your decision? Was Story A created by modifying Story 1 or Story 2? Respond with 1 if it was created by modifying Story 1, and 2 if it was created by modifying Story 2."""

    return prompt_1, prompt_2

def gpt4matcher(response_1, response_2, perturbed):
    prompt_1, prompt_2 = generate_comparison_prompts(response_1, response_2, perturbed)

    for _ in range(3):
        first, second = query_openai_with_history(prompt_1, prompt_2)
        log.info(f"Model's First Response: {first.content}")
        log.info(f"Model's Second Response: {second.content}")
        log.info("---------------------------------------------------------------")
    
        decision = second.content[0]

        if (decision == "1") or (decision == "2"):
            log.info(f"Valid decision made: {decision}")
            return int(decision)

    # If no valid decision is made after 3 trials, return 0
    return 0

def distinguish(matcher ,response_1, response_2, perturbed, num_repetitions):
    """
    matcher is a function like the one above.
    """
    regular_match = lambda: matcher(response_1, response_2, perturbed)

    # Adjust the response of the flipped match function using cool functional programming
    handle_flipped = lambda x: 2 if x == 1 else 1 if x == 2 else x
    flipped_match = lambda: handle_flipped(matcher(response_2, response_1, perturbed))

    half_repetitions = int(num_repetitions/2)

    decisions = [func() for i in range(half_repetitions) for func in (regular_match, flipped_match)]
    log.info("Decisions recorded: %s", decisions)

    successful_decisions = [d for d in decisions if d in [1, 2]]
    num_successful_attempts = len(successful_decisions)

    decision_count = Counter(successful_decisions)
    log.info("Decision count: %s", decision_count)
    
    # We multiply by 2 here since the number of repetitions is actually 2 * num_repetitions.
    # TODO: Change the threshold 0.5 to be a parameter if necessary.
    threshold = int(num_successful_attempts * 0.5)

    log.info(f"Threshold: {threshold}")
    log.info(f"Number of 1's: {decision_count[1]}")
    
    return 1 if decision_count[1] >= threshold else 2

def get_response_and_perturbed(entropy, output_num, attack_id, mutation_num):
    csv_file_directory = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/"
    
    first_perturbed_csv_filename = f"output_{output_num}/corpuses/attack_{attack_id}.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    response = get_watermarked_text(csv_file_path)
    perturbed = get_nth_successful_perturbation(csv_file_path, mutation_num)

    return response, perturbed

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Construct log filename based on command line arguments
    log_filename = f"./results/stationary_distribution/robustness_analysis/entropy_{cfg.distinguisher.entropy}/distinguisher_results/{cfg.distinguisher.output_1}_{cfg.distinguisher.attack_id_1}-{cfg.distinguisher.output_2}_{cfg.distinguisher.attack_id_2}_{cfg.distinguisher.log_suffix}.log"

    # Configure logging to dynamically created filename
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')
    
    # Log script arguments
    # TODO: You can already read this from Hydra actually.
    log.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Old distinguisher used the prompt, so these are here.
    # prompts_file_path = './inputs/dynamic_prompts.csv'
    # prompt = get_prompt_or_output(prompts_file_path, args.entropy) 
    
    # Get Perturbed Versions
    response_1, perturbed_1 = get_response_and_perturbed(cfg.distinguisher.entropy, cfg.distinguisher.output_1, cfg.distinguisher.attack_id_1, cfg.distinguisher.mutation_num)
    response_2, perturbed_2 = get_response_and_perturbed(cfg.distinguisher.entropy, cfg.distinguisher.output_2, cfg.distinguisher.attack_id_2, cfg.distinguisher.mutation_num)

    if cfg.distinguisher.matcher == "gpt4":
        matcher = gpt4matcher
    # TODO: Only works with Llama right now.
    elif cfg.distinguisher.matcher == "local":
        distinguisher_object = Distinguisher(cfg.distinguisher.distinguisher_args)
        matcher = distinguisher_object.match

    log.info(f"Num repetitions: {cfg.distinguisher.num_repetitions}")

    distinguisher = partial(distinguish, matcher, response_1, response_2, num_repetitions=cfg.distinguisher.num_repetitions)

    run_trials = lambda perturbed, answer: [distinguisher(perturbed) == answer for _ in range(cfg.distinguisher.num_trials)]

    perturbed_1_trials = run_trials(perturbed_1, 1)
    perturbed_2_trials = run_trials(perturbed_2, 2)

    perturbed_1_success_rate = float(sum(perturbed_1_trials))/cfg.distinguisher.num_trials
    perturbed_2_success_rate = float(sum(perturbed_2_trials))/cfg.distinguisher.num_trials

    log.info("Perturbed 1 success rate: %.2f%%", perturbed_1_success_rate * 100)
    log.info("Perturbed 2 success rate: %.2f%%", perturbed_2_success_rate * 100)

    
if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()