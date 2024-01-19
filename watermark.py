import torch
import numpy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, pipeline
# UMD
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
# UNIGRAM
from gptwm import GPTWatermarkLogitsWarper, GPTWatermarkDetector
# EXP
from exp_generate import generate_shift
from exp_detect import permutation_test

# For testing
import hydra
from omegaconf import DictConfig, OmegaConf

class Watermarker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.generator_args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, cache_dir = cfg.model_cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=cfg.model_cache_dir,
            device_map=cfg.generator_args.device_map,
            trust_remote_code=False,
            revision=cfg.generator_args.revision)
            
        self.watermarking_scheme = cfg.watermark_args.name
        
        
        # Initialize the watermark processor and detector objects
        if self.watermarking_scheme == "umd":
            self.watermark_processor = WatermarkLogitsProcessor(vocab=list(self.tokenizer.get_vocab().values()),
                                                        gamma=0.25,
                                                        delta=2.0,
                                                        seeding_scheme="selfhash")
            
            self.watermark_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=0.25,
                                        seeding_scheme="selfhash",
                                        device=cfg.generator_args.device_map,
                                        tokenizer=self.tokenizer,
                                        z_threshold=cfg.watermark_args.z_threshold,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
            
        elif self.watermarking_scheme == "unigram":
            self.watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=0.5,
                                                                        strength=2.0,
                                                                        vocab_size=self.tokenizer.vocab_size,
                                                                        watermark_key=cfg.watermark_args.watermark_key)])
            self.watermark_detector = GPTWatermarkDetector(fraction=0.5,
                                        strength=2.0,
                                        vocab_size=self.tokenizer.vocab_size,
                                        watermark_key=0)
        elif self.watermarking_scheme == "exp":
            self.p_threshold = cfg.watermark_args.p_threshold
            
        # # Store the pipeline configuration
        self.generator_kwargs = {
            "max_new_tokens": cfg.generator_args.max_new_tokens,
            "do_sample": cfg.generator_args.do_sample,
            "temperature": cfg.generator_args.temperature,
            "top_p": cfg.generator_args.top_p,
            "top_k": cfg.generator_args.top_k,
            "repetition_penalty": cfg.generator_args.repetition_penalty
        }
        
        # Additional parameters based on watermarking scheme
        if self.watermarking_scheme == "umd":
            self.generator_kwargs["logits_processor"] = LogitsProcessorList([self.watermark_processor])
        elif self.watermarking_scheme == "unigram":
            self.generator_kwargs["logits_processor"] = self.watermark_processor
            self.generator_kwargs["output_scores"] = True
        elif self.watermarking_scheme == "exp":
            torch.manual_seed(cfg.watermark_args.seed)
            self.watermark_sequence_length = cfg.watermark_args.watermark_sequence_length
            self.generated_text_length = cfg.watermark_args.generated_text_length
            self.watermark_sequence_key = cfg.watermark_args.watermark_sequence_key
    
        self.is_completion = cfg.generator_args.is_completion
            
    def generate(self):
        n_attemps = 0
        while n_attemps < 5:
            # TODO: Add a check here to see if the completion was successful and a retry mechanism to handle unsuccessful completions.
            if self.watermarking_scheme == "exp":
                tokens = self.tokenizer.encode(self.cfg.generator_args.prompt, return_tensors='pt', truncation=True, max_length=2048)
                outputs = generate_shift(self.model,tokens,len(self.tokenizer),self.watermark_sequence_length,self.generated_text_length,self.watermark_sequence_key)
            else:
                # completion = self.pipe(prompt)[0]['generated_text']
                inputs = self.tokenizer(self.cfg.generator_args.prompt, return_tensors="pt", padding=True, truncation=True, max_length=100).to(self.model.device)
                outputs = self.model.generate(**inputs, **self.generator_kwargs)
                    
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Get rid of the prompt
            if not self.is_completion:
                completion = completion.replace(self.cfg.generator_args.prompt, '', 1).strip()
            
            print(completion)
            print(len(completion))
            
            if len(completion) > 1500:
                break
            
            n_attempts += 1
            
        return completion
        
    def detect(self, completion):
        if self.watermarking_scheme == "umd":  
            score = self.watermark_detector.detect(completion)
            score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
            z_score = score_dict['z_score']
            prediction = score_dict['prediction']
            return prediction, z_score
        elif self.watermarking_scheme == "unigram":
            token_sequence = self.tokenizer(completion, add_special_tokens=False)['input_ids']
            z_score = self.watermark_detector.detect(token_sequence, device=self.model.device)
            prediction = (z_score >= self.cfg.watermark_args.z_threshold)
            return prediction, z_score
        elif self.watermarking_scheme == "exp":
            tokens = self.tokenizer.encode(completion, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
            pval = permutation_test(tokens,self.watermark_sequence_key,self.watermark_sequence_length,len(tokens),len(self.tokenizer))
            prediction = (pval <= self.p_threshold)
            return prediction, pval
        return None
        

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    watermarker = Watermarker(cfg)
    
    import time
    for _ in range(2):
        completion = watermarker.generate()
        
        print(completion)
        
        start = time.time()
        score, prediction = watermarker.detect(completion)
        delta = time.time() - start
        
        print(f"Score: {score}")
        print(f"Prediction: {prediction}")
        print (f"Time taken: {delta}")
         
            
if __name__ == "__main__":
    main()