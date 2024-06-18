import subprocess
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

log = logging.getLogger(__name__)

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


# NOTE: pass in as an argument the prompt number (1, 2, or 3)
# this was so i could parallelize it better with gpus
def main():
    base_folder_name = './inputs/test_tokenizer_prompt_based_saves'

    results = []
    temps = [0.5, 1, 1.5, 1.8]
    divps = [0, 10, 20]
    

    prompt_num = int(sys.argv[1])

    for attempt in range(1, 4):
        for temp in temps:
            for divp in divps:
                folder_name = f'prompt_{prompt_num}_temp_{int(temp * 100)}_divp_{divp}_attempt_{attempt}'
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname, f'{base_folder_name}/{folder_name}/')
                if not os.path.exists(path):
                    os.makedirs(path)

                log_filepath = f"{base_folder_name}/{folder_name}/logfile.log"
                
                # TODO: The is_completion arg needs to be fixed.
                command = f"python watermarked_text_generator.py " \
                        f"++prompt_file='./inputs/tests_v1_with_lotr.csv' " \
                        f"++prompt_num={prompt_num} " \
                        f"++attack_args.is_completion=False " \
                        f"++is_completion=False " \
                        f"++generator_args.temperature={temp} " \
                        f"++generator_args.diversity_penalty={divp} " \
                        f"++generation_stats_file_path='{base_folder_name}/{folder_name}/stats.csv' " \
                        f"++watermarked_text_file_name='new_prompt_based_saves/{folder_name}/watermarked_text.csv' "

                
                stdout, stderr = run_command(command, log_filepath)
                if stderr is None:
                    print(f"Command succeeded: {command}\nOutput:\n{stdout}")
                else:
                    print(f"Command failed: {command}\nError:\n{stderr}")
                results.append((stdout, stderr))

    log.info(results)

if __name__ == "__main__":
    main()
