import logging
import hydra
from utils import get_watermarker, save_to_csv, get_prompt_or_output, count_csv_entries

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):
    import time
    import textwrap
    
    log.info(f"Starting to watermark...")

    # Read the prompt and the watermarked text from the input files
    prompt = cfg.attack_args.prompt
    if prompt is None:
        prompt = get_prompt_or_output(cfg.attack_args.prompt_file, cfg.attack_args.prompt_num) 

    log.info(f"Prompt: {prompt}")

    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    for _ in range(1):
        start = time.time()
        watermarked_text = watermarker.generate(prompt)
        is_detected, score = watermarker.detect(watermarked_text)
        delta = time.time() - start
        
        log.info(f"Watermarked Text: {watermarked_text}")
        log.info(f"Is Watermark Detected?: {is_detected}")
        log.info(f"Score: {score}")
        log.info(f"Time taken: {delta}")

    if cfg.watermark_args.save_file_name is not None:
        file_path = f"./inputs/{cfg.watermark_args.save_file_name}"
        num_entries = count_csv_entries(file_path)

        stats = [{'num': num_entries +1, 'text': watermarked_text}]
        save_to_csv(stats, './inputs', cfg.watermark_args.save_file_name)

if __name__ == "__main__":
    test()
