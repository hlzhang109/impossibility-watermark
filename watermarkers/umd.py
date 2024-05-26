import logging
from watermarker import Watermarker

import torch
from transformers import LogitsProcessorList

# UMD
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

log = logging.getLogger(__name__)

class UMDWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, is_completion=False):
        super().__init__(cfg, pipeline, n_attempts, is_completion)

    def _setup_watermark_components(self):
        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=self.cfg.watermark_args.gamma,
            delta=self.cfg.watermark_args.delta,
            seeding_scheme=self.cfg.watermark_args.seeding_scheme
        )
        
        self.watermark_detector = WatermarkDetector(
            tokenizer=self.tokenizer,
            vocab=list(self.tokenizer.get_vocab().values()),
            z_threshold=self.cfg.watermark_args.z_threshold,
            gamma=self.cfg.watermark_args.gamma,
            seeding_scheme=self.cfg.watermark_args.seeding_scheme,
            normalizers=self.cfg.watermark_args.normalizers,
            ignore_repeated_ngrams=self.cfg.watermark_args.ignore_repeated_ngrams,
            device=self.cfg.watermark_args.device,
        )
        
        self.generator_kwargs["logits_processor"] = LogitsProcessorList([self.watermark_processor])

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
        score = self.watermark_detector.detect(completion)
        score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
        z_score = score_dict['z_score']
        is_detected = score_dict['prediction']
        return is_detected, z_score