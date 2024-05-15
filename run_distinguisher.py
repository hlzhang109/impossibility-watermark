import subprocess

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

log = logging.getLogger(__name__)

attack_1 = {
    "entropy": 1,
    "output_1": 1,
    "attack_id_1" : 5,
    "output_2": 3,
    "attack_id_2": 1,
}

attack_4 = {
    "entropy": 4,
    "output_1": "1",
    "attack_id_1" : "2_1",
    "output_2": "2",
    "attack_id_2": "1_1",
}

attack_5 = {
    "entropy": 5,
    "output_1": 1,
    "attack_id_1" : "1_1",
    "output_2": 2,
    "attack_id_2": "1_1",
}

attack_6 = {
    "entropy": 6,
    "output_1": 1,
    "attack_id_1" : "2",
    "output_2": 2,
    "attack_id_2": "2",
}

# good_attacks = [attack_1, attack_4, attack_5, attack_6]
# good_attacks = [attack_1]
# good_attacks = [attack_4]
good_attacks = [attack_5, attack_6]

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

def main():
    log_suffix = "first_experiments"
    results = []

    for attack in good_attacks:
        log_filepath = f"./results/stationary_distribution/robustness_analysis/entropy_{attack['entropy']}/distinguisher_results/{attack['output_1']}_{attack['attack_id_1']}-{attack['output_2']}_{attack['attack_id_2']}_{log_suffix}.log"

        command = f"python distinguisher.py +distinguisher=stingy_gpt4 ++distinguisher.log_suffix={log_suffix} " \
                  f"++distinguisher.entropy='\"{attack['entropy']}\"' " \
                  f"++distinguisher.output_1='\"{attack['output_1']}\"' " \
                  f"++distinguisher.attack_id_1='\"{attack['attack_id_1']}\"' " \
                  f"++distinguisher.output_2='\"{attack['output_2']}\"' " \
                  f"++distinguisher.attack_id_2='\"{attack['attack_id_2']}\"' "
        stdout, stderr = run_command(command, log_filepath)
        if stderr is None:
            print(f"Command succeeded: {command}\nOutput:\n{stdout}")
        else:
            print(f"Command failed: {command}\nError:\n{stderr}")
        results.append((stdout, stderr))

    log.info(results)

if __name__ == "__main__":
    main()
