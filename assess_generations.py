from oracles.absolute import PrometheusAbsoluteOracle
from utils import get_prompt_or_output

def main():
    oracle = PrometheusAbsoluteOracle()
  
    import os
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    log = logging.getLogger(__name__)

    prompt_nums = [1, 2, 3]

    # prompt_based_or_c4 = "prompt_based_saves" 
    prompt_based_or_c4 = "c4_saves"

    # TODO: This is pretty dumb. We wanna have the same settings for both.
    if prompt_based_or_c4 == "prompt_based_saves":
        temps = [1, 1.3, 1.7]
        divps = [5, 15, 20]
    else:
        temps = [0.5, 1, 1.5, 1.8]
        divps = [0,15,20]


    for attempt in range(1, 4):
        for temp in temps:
            for divp in divps:
                for prompt_num in prompt_nums:
                    if prompt_based_or_c4 == "prompt_based_saves":
                        folder_name = f'prompt_{prompt_num}_temp_{int(temp * 100)}_divp_{divp}_attempt_{attempt}'
                    else:
                        folder_name = f'c4_{prompt_num}_temp_{int(temp * 100)}_divp_{divp}_attempt_{attempt}'
                    input_folder_name = f'./inputs/{prompt_based_or_c4}/{folder_name}'

                    # NOTE: Only works with prompt_based right now.
                    if prompt_based_or_c4 == "prompt_based_saves":
                        prompt_path = './inputs/tests_v1_with_lotr.csv'
                        prompt = get_prompt_or_output(prompt_path, prompt_num)
                    else:
                        prompt_path = './inputs/mini_c4.csv'
                        prefix = get_prompt_or_output(prompt_path, prompt_num)
                        prompt = f"Complete the following prefix in a coherent and creative manner. You can complete it in any way you like.\nPrefix:\n{prefix}"

                    log.info(f"Prompt\n {prompt}")

                    watermarked_text_path = f"{input_folder_name}/watermarked_text_without_commas.csv"
                    watermarked_text_num = 1

                    watermarked_text = get_prompt_or_output(watermarked_text_path, watermarked_text_num)

                    log.info(f"Watermarked Text\n {watermarked_text}")

                    log_filepath = f"./inputs/{prompt_based_or_c4}/{folder_name}/quality_logfile_without_commas.log"
                    log.info(f"Running Absolute Oracle with {folder_name}")
									    
                    try:
                        with open(log_filepath, "w") as f:
                            print("Evaluation WITHOUT Reference Answer")
                            feedback, score = oracle.evaluate(prompt, watermarked_text, None)
                            print("Feedback:", feedback, file=f)
                            print("Score:", score, file=f)

                        log.info("Oracle executed successfully")
                    except Exception as e:
                        log.error(f"Oracle failed with error: {e}")
                  
                    log.info(f"Saving results to {log_filepath}")

if __name__ == "__main__":
    main()