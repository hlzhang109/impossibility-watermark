import logging
import hydra
from utils import get_watermarker

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):
    import time
    import textwrap
    
    log.info(f"Starting to watermark...")

    prompt = textwrap.dedent(
        """Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings 
        series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?
        
        Answer:"""
    )

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

if __name__ == "__main__":
    test()
