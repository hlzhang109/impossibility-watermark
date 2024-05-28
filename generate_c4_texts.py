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
    results = []

    for prompt_num in range(1,11):

        log_filepath = f"./logs/c4/minic4_{prompt_num}_second.log"
        
        command = f"python watermarked_text_generator.py " \
                  f"++attack_args.prompt_file='./inputs/mini_c4.csv' " \
                  f"++attack_args.prompt_num={prompt_num} " \
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
