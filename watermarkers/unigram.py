import logging
from watermarker import Watermarker

import torch
from transformers import LogitsProcessorList

from gptwm import GPTWatermarkLogitsWarper, GPTWatermarkDetector

log = logging.getLogger(__name__)

class UnigramWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, is_completion=False):
        super().__init__(cfg, pipeline, n_attempts, is_completion)

    def _setup_watermark_components(self):
        assert self.tokenizer.vocab_size == self.model.config.vocab_size, f"Tokenizer vocab size {self.tokenizer.vocab_size} does not match model vocab size {self.model.config.vocab_size}"

        self.watermark_processor = LogitsProcessorList([
            GPTWatermarkLogitsWarper(
                fraction=self.cfg.watermark_args.fraction,
                strength=self.cfg.watermark_args.strength,
                watermark_key=self.cfg.watermark_args.watermark_key,
                vocab_size=self.tokenizer.vocab_size,)])
        
        self.watermark_detector = GPTWatermarkDetector(
            fraction=self.cfg.watermark_args.fraction,
            strength=self.cfg.watermark_args.strength,
            watermark_key=self.cfg.watermark_args.watermark_key,
            vocab_size=self.tokenizer.vocab_size,)

        # Adjust the generator kwargs for unigram watermarking
        self.generator_kwargs["logits_processor"] = self.watermark_processor
        self.generator_kwargs["output_scores"] = True

    def generate_watermarked_outputs(self, prompt):
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.cfg.generator_args.max_new_tokens
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, **self.generator_kwargs)

        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion

    def detect(self, completion):
        token_sequence = self.tokenizer(completion, add_special_tokens=False)['input_ids']
        z_score = self.watermark_detector.detect(token_sequence, device=self.model.device)
        is_detected = (z_score >= self.cfg.watermark_args.z_threshold)
        return is_detected, z_score
