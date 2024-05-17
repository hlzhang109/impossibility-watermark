import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

log = logging.getLogger(__name__)

def generate_attack_dictionaries(entropy_value, entropy_dict):
    attack_dictionaries = []
    keys = list(entropy_dict.keys())

    for idx, key in enumerate(keys):
        other_keys = keys[idx+1:]  # Only use keys that come after the current key
        for other_key in other_keys:
            for key_attack_id in entropy_dict[key]:
                for other_key_attack_id in entropy_dict[other_key]:
                    attack_dict = {
                        "entropy": entropy_value,
                        "output_1" : key,
                        "attack_id_1" : key_attack_id,
                        "output_2" : other_key,
                        "attack_id_2": other_key_attack_id,
                    }
                    attack_dictionaries.append(attack_dict)
                    
    return attack_dictionaries

entropy_1 = {
    "1": ["3_1", "5"],
    "2": ["2_1", "4"],
    "3": ["1", "2", "3"]
}

entropy_4 = {
    "1": ["2_1", "3_1"],
    "2": ["1_1"],
    "3": ["2", "4"]
}

entropy_5 = {
    "1": ["1_1", "2_1", "3_1"],
    "2": ["1_1", "2_1", "3_1"],
}

entropy_6 = {
    "1": ["2", "3"],
    "2": ["2", "3", "4"],
}


good_attacks = []
entropy_1_attacks = generate_attack_dictionaries(1, entropy_1)
entropy_4_attacks = generate_attack_dictionaries(4, entropy_4)
entropy_5_attacks = generate_attack_dictionaries(5, entropy_5)
entropy_6_attacks = generate_attack_dictionaries(6, entropy_6)

good_attacks += entropy_1_attacks 
# good_attacks += entropy_4_attacks
# good_attacks += entropy_5_attacks 
# good_attacks += entropy_6_attacks

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
    log_suffix = "05_16_llama_mass_huge_batch"
    results = []

    for attack in good_attacks:
        log_filepath = f"./results/stationary_distribution/robustness_analysis/entropy_{attack['entropy']}/distinguisher_results/{attack['output_1']}_{attack['attack_id_1']}-{attack['output_2']}_{attack['attack_id_2']}_{log_suffix}.log"

        command = f"python distinguisher.py +distinguisher=llama3 ++distinguisher.log_suffix={log_suffix} " \
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
