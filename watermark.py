import torch
from transformers import AutoTokenizer
# UMD
from extended_watermark_processor import WatermarkDetector
# UNIGRAM
from gptwm import GPTWatermarkDetector

class Watermark:
    def __init__(self, args):
        self.generative_model_name = args.generative_model_name
        self.generative_model_device = args.generative_model_device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.watermarking_scheme = args.watermarking_scheme
        self.z_threshold = args.z_threshold
        
        if self.watermarking_scheme == "umd":
            self.watermark_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=0.25,
                                        seeding_scheme="selfhash",
                                        device=self.generative_model_device,
                                        tokenizer=self.tokenizer,
                                        z_threshold=args.z_threshold,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
        elif self.watermarking_Scheme == "unigram":
            # TODO: Parametrize the wm_key.
            self.watermark_detector = GPTWatermarkDetector(fraction=0.5,
                                        strength=2.0,
                                        vocab_size=self.tokenizer.vocab_size,
                                        watermark_key=0)
        
        
    def detect(self, completion):
        if self.watermarking_scheme == "umd":  
            score = self.watermark_detector.detect(completion)
            score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
            z_score = score_dict['z_score']
            prediction = score_dict['prediction']
        elif self.watermarking_scheme == "unigram":
            token_sequence = self.tokenizer(completion, add_special_tokens=False)['input_ids']
            z_score = self.watermark_detector.detect(token_sequence, device=self.generative_model_device)
            prediction = (z_score >= self.z_threshold)
        return prediction, z_score

            