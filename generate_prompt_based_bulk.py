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


# pass in as an argument the prompt number (1, 2, or 3)
# this was so i could parallelize it better with gpus
def main():
    results = []
    temps = [1, 1.3, 1.7]
    divps = [5, 15, 20]
    

    prompt_num = int(sys.argv[1])

    for attempt in range(1, 4):
        for temp in temps:
            for divp in divps:
                folder_name = f'prompt_{prompt_num}_temp_{int(temp * 100)}_divp_{divp}_attempt_{attempt}'
                dirname = os.path.dirname(__file__)
                path = os.path.join(dirname, f'./inputs/prompt_based_saves/{folder_name}/')
                if not os.path.exists(path):
                    os.makedirs(path)

                log_filepath = f"./inputs/prompt_based_saves/{folder_name}/logfile.log"
                
                command = f"python watermarked_text_generator.py " \
                        f"++attack_args.prompt_file='./inputs/tests_v1_with_lotr.csv' " \
                        f"++attack_args.prompt_num={prompt_num} " \
                        f"++attack_args.is_completion=False " \
                        f"++generator_args.temperature={temp} " \
                        f"++generator_args.diversity_penalty={divp} " \
                        f"++generator_args.generation_stats_csv_path='./inputs/prompt_based_saves/{folder_name}/stats.csv' " \
                        f"++watermark_args.save_file_name='prompt_based_saves/{folder_name}/watermarked_text.csv' "

                
                stdout, stderr = run_command(command, log_filepath)
                if stderr is None:
                    print(f"Command succeeded: {command}\nOutput:\n{stdout}")
                else:
                    print(f"Command failed: {command}\nError:\n{stderr}")
                results.append((stdout, stderr))

    log.info(results)

if __name__ == "__main__":
    main()
