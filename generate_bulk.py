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
            _ = subprocess.run(command, shell=True, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
        log.info("Command executed successfully")
        return "Success", None
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")
        # Returning None, stderr if there was an error
        return None, e.stderr


# NOTE: pass in as an argument the prompt number (1, 2, or 3)
# this was so i could parallelize it better with gpus
def main():
    dir_name = "newest_prompt_based_saves"
    base_folder_name = f'./inputs/{dir_name}'

    results = []
    # temps = [0.5, 1, 1.5, 1.8]
    # divps = [0, 10, 20]
    # temps = [1]
    # divps = [0]
    temps = [1, 1.5, 1.8]
    divps = [0,10,20]
    
    # prompt_num = int(sys.argv[1])

    # NOTE: We might want to turn this into a command line argument.
    gen_type = "prompt" # "c4"

    if gen_type not in ["prompt", "c4"]:
        raise Exception(f"Generation type {gen_type} is not supported.")

    for prompt_num in range(7,10):
        for attempt in range(1, 4):
            for temp in temps:
                for divp in divps:
                    if gen_type == "prompt":
                        folder_name = f'prompt_{prompt_num}_temp_{int(temp * 100)}_divp_{divp}_attempt_{attempt}'
                    elif gen_type == "c4":
                        folder_name = f'c4_{prompt_num}_temp_{int(temp * 100)}_divp_{divp}_attempt_{attempt}'

                    dirname = os.path.dirname(__file__)
                    path = os.path.join(dirname, f'{base_folder_name}/{folder_name}/')
                    if not os.path.exists(path):
                        os.makedirs(path)

                    log_filepath = f"{base_folder_name}/{folder_name}/logfile.log"
                    
                    if gen_type == "prompt":
                        command = f"python watermarked_text_generator.py " \
                                f"++prompt_file='./inputs/tests_v1_with_lotr.csv' " \
                                f"++prompt_num={prompt_num} " \
                                f"++is_completion=False " \
                                f"++generator_args.temperature={temp} " \
                                f"++generator_args.diversity_penalty={divp} " \
                                f"++generation_stats_file_path='{base_folder_name}/{folder_name}/stats.csv' " \
                                f"++watermark_args.use_fine_tuned=False " \
                                f"++watermarked_text_file_name='{dir_name}/{folder_name}/watermarked_text.csv' "
                    elif gen_type == "c4":
                        command = f"python watermarked_text_generator.py " \
                                f"++prompt_file='./inputs/mini_c4.csv' " \
                                f"++prompt_num={prompt_num} " \
                                f"++is_completion=True " \
                                f"++generator_args.temperature={temp} " \
                                f"++generator_args.diversity_penalty={divp} " \
                                f"++generation_stats_file_path='{base_folder_name}/{folder_name}/stats.csv' " \
                                f"++watermark_args.use_fine_tuned=True " \
                                f"++watermarked_text_file_name='{dir_name}/{folder_name}/watermarked_text.csv' "
    
                    stdout, stderr = run_command(command, log_filepath)
                    if stderr is None:
                        print(f"Command succeeded: {command}\nOutput:\n{stdout}")
                    else:
                        print(f"Command failed: {command}\nError:\n{stderr}")
                    results.append((stdout, stderr))

    log.info(results)

if __name__ == "__main__":
    main()