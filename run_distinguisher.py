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
    "output_1": 1,
    "attack_id_1" : "2_1",
    "output_2": 2,
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
good_attacks = [attack_1]

def run_command(command):
    log.info(f"Running command: {command}")

    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, e.stderr

def main():
    log_suffix = "first_experiments"
    results = []

    for attack in good_attacks:
        log_filename = f"./results/stationary_distribution/robustness_analysis/entropy_{attack['entropy']}/distinguisher_results/{attack['output_1']}_{attack['attack_id_1']}-{attack['output_2']}_{attack['attack_id_2']}_{log_suffix}.log"

        command = f"python optimized_distinguisher.py +distinguisher=stingy_gpt4 ++distinguisher.log_suffix={log_suffix} " \
                  f"++distinguisher.entropy={attack['entropy']} " \
                  f"++distinguisher.output_1={attack['output_1']} " \
                  f"++distinguisher.attack_id_1={attack['attack_id_1']} " \
                  f"++distinguisher.output_2={attack['output_2']} " \
                  f"++distinguisher.attack_id_2={attack['attack_id_2']}" \
                  f" &> {log_filename}"
        stdout, stderr = run_command(command)
        if stderr is None:
            print(f"Command succeeded: {command}\nOutput:\n{stdout}")
        else:
            print(f"Command failed: {command}\nError:\n{stderr}")
        results.append((stdout, stderr))

    log.info(results)

if __name__ == "__main__":
    main()
