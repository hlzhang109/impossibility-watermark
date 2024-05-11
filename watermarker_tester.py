import logging
import hydra

log = logging.getLogger(__name__)

# TODO: Can be dynamic imports in principle.
from umd import UMDWatermarker
from unigram import UnigramWatermarker
from exp import EXPWatermarker

def get_watermarker(cfg):
    if cfg.watermark_args.name == "umd":
        return UMDWatermarker(cfg)
    elif cfg.watermark_args.name == "unigram":
        return UnigramWatermarker(cfg)
    elif cfg.watermark_args.name == "exp":
        return EXPWatermarker(cfg)
    else:
        raise

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):
    import time
    import textwrap
    
    prompt = textwrap.dedent(
        """Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings 
        series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?"""
    )

    watermarker = get_watermarker(cfg)

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
