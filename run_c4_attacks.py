import subprocess
import logging

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

def main():
    # TODO: Adjust to new config.
    results = []

    for prompt_num in range(1,2):
        log_filepath = f"./logs/semstamp_attacks/minic4_{prompt_num}_first.log"
        
        # NOTE: Change the save_file_name to a new filename every time you run the code.
        command = f"python attack.py " \
                  f"++attack_args.prompt_file='./inputs/mini_c4.csv' " \
                  f"++attack_args.prompt_num={prompt_num} " \
                  f"++attack_args.watermarked_text_path='./inputs/c4_{prompt_num}.csv' " \
                  f"++attack_args.watermarked_text_num=1 " \
                  f"++attack_args.results_dir='./semstamp_attacks/c4/' " \
                  f"++watermark_args.save_file_name='c4_{prompt_num}.csv' " \
                  f"++attack_args.is_completion=True " 

        stdout, stderr = run_command(command, log_filepath)
        if stderr is None:
            print(f"Command succeeded: {command}\nOutput:\n{stdout}")
        else:
            print(f"Command failed: {command}\nError:\n{stderr}")
        results.append((stdout, stderr))

    log.info(results)

if __name__ == "__main__":
    main()
