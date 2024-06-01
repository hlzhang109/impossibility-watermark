import subprocess
import logging
import hydra

from attack import Attack
from utils import get_prompt_or_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger(__name__)

import re

def remove_double_triple_commas(text):
    # Remove triple commas first
    text = re.sub(r',,,', ',', text)
    # Then remove double commas
    text = re.sub(r',,', ',', text)
    return text

def run_command(command, filepath):
    log.info(f"Running command: {command}")
    log.info(f"Saving results to {filepath}")

    try:
        with open(filepath, "w") as f:
            # Redirect both stdout and stderr to the same file
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
        log.info("Command executed successfully")
        return "Success", None
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")
        # Returning None, stderr if there was an error
        return None, e.stderr

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # TODO: Adjust the prompt file and prompt num.

    prompt_path = './inputs/tests_v1_with_lotr.csv'
    prompt_num = 3

    prompt = get_prompt_or_output(prompt_path, prompt_num)


    watermarked_text_path = "/local1/borito1907/impossibility-watermark/inputs/prompt_based_saves/prompt_3_temp_100_divp_20_attempt_1/watermarked_text.csv"
    watermarked_text_num = 1

    watermarked_text = get_prompt_or_output(watermarked_text_path, watermarked_text_num)

    # TODO: There should be a better way to do this.
    watermarked_text = remove_double_triple_commas(watermarked_text)

    log.info(f"Prompt: {prompt}")
    log.info(f"Watermarked Text: {watermarked_text}")

    cfg.attack_args.log_csv_path = './semstamp_attacks/05_31_prompt_based_lotr.csv'

    attacker = Attack(cfg)
    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
