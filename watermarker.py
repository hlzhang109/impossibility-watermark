from abc import ABC, abstractmethod
import torch
import logging
import hydra

from pipeline import PipeLineBuilder

# TODO: Can be dynamic imports in principle.
from umd import UMDWatermarker
from unigram import UnigramWatermarker
from exp import EXPWatermarker

log = logging.getLogger(__name__)

class Watermarker(ABC):
    def __init__(self, cfg, pipeline=None, n_attempts=10, is_completion=False):
        self.cfg = cfg # config.watermark_args
        self.n_attempts = n_attempts
        self.pipeline = pipeline
        self.is_completion = is_completion
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        log.info(f"Using device: {self.device}")

        if not isinstance(self.pipeline, PipeLineBuilder):
            self.pipeline = PipeLineBuilder(self.cfg.generator_args)
        
        self.model = self.pipeline.model.to(self.device)
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        self.generator_kwargs = {
            "max_new_tokens": self.cfg.generator_args.max_new_tokens,
            "do_sample": self.cfg.generator_args.do_sample,
            "temperature": self.cfg.generator_args.temperature,
            "top_p": self.cfg.generator_args.top_p,
            "top_k": self.cfg.generator_args.top_k,
            "repetition_penalty": self.cfg.generator_args.repetition_penalty
        }
        
        self.setup_generator_kwargs()

    @abstractmethod
    def setup_watermark_components(self):
        pass

    @abstractmethod
    def generate_watermarked_outputs(self, prompt):
        pass

    def generate(self, prompt):
        n_attempts = 0
        while n_attempts < self.n_attempts:
            outputs = self.generate_watermarked_outputs(prompt)

            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not self.is_completion:
                completion = completion.replace(prompt, '', 1).strip()

            # Check if watermark succeeded
            _, p_value = self.detect(completion)
            if p_value <= self.p_threshold:
                return completion
            else:
                log.info("Failed to watermark, trying again...")
                n_attempts += 1

        return None

    @abstractmethod
    def detect(self, completion):
        pass

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


