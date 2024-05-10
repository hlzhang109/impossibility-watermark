from abc import ABC, abstractmethod
import torch
import logging

from pipeline import PipeLineBuilder

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
        
        self.setup_watermark_components()

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




