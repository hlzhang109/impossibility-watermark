import logging
from watermarker import Watermarker

import torch
from transformers import LogitsProcessorList

# EXP
from exp_generate import generate_shift
from exp_detect import permutation_test

log = logging.getLogger(__name__)

class EXPWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, is_completion=False):
        super().__init__(cfg, pipeline, n_attempts, is_completion)

    def setup_watermark_components(self):
        torch.manual_seed(self.cfg.watermark_args.seed)
        self.watermark_sequence_length = self.cfg.watermark_args.watermark_sequence_length
        self.generated_text_length = self.cfg.watermark_args.generated_text_length
        self.watermark_sequence_key = self.cfg.watermark_args.watermark_sequence_key
        self.p_threshold = self.cfg.watermark_args.p_threshold

    def generate_watermarked_outputs(self, prompt):
        tokens = self.tokenizer.encode(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.cfg.generator_args.max_new_tokens
        ).to(self.model.device)

        outputs = generate_shift(
            model=self.model,
            prompt=tokens,
            vocab_size=len(self.tokenizer),
            n=self.watermark_sequence_length,
            m=self.generated_text_length,
            key=self.watermark_sequence_key
        )

        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return completion

    def detect(self, completion):
        tokens = self.tokenizer.encode(completion, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
        pval = permutation_test(tokens, self.watermark_sequence_key, self.watermark_sequence_length, len(tokens), len(self.tokenizer))
        is_detected = (pval <= self.p_threshold)
        return is_detected, pval
