from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, pipeline

# UMD
from extended_watermark_processor import WatermarkLogitsProcessor
# Unigram
from gptwm import GPTWatermarkLogitsWarper

class TextGenerator:
    # TODO: Add C4 dataset support.
    # TODO: Add EXP scheme.
    def __init__(self, cfg):
        self.watermarking_scheme = cfg.name
                
        print(f"Text Generation Model: {self.model_name_or_path}")
        print(f"Watermarking Scheme: {self.watermarking_scheme}")
        
        # Initialize and load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path=cfg.model_name_or_path,
            cache_dir=cfg.model_cache_dir,
            device_map=cfg.device_map,
            trust_remote_code=False,
            revision=cfg.revision) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, use_fast=True, cache_dir=cfg.model_cache_dir)
        
        # Store the pipeline configuration
        self.generation_config = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty
        }

        # Additional parameters based on watermarking scheme
        if self.watermarking_scheme == "umd":
            self.generation_config["logits_processor"] = LogitsProcessorList([self.watermark_processor])
        elif self.watermarking_scheme == "unigram":
            self.generation_config["logits_processor"] = self.watermark_processor
            self.generation_config["output_scores"] = True

        # Create the pipeline
        self.pipe = pipeline("text-generation", **self.pipeline_config)
        
        # TODO: Add completion vs. prompt generation.
        self.is_completion = False
    
    def generate(self, prompt, num_samples):
        results = []
        successful_generations = 0
        while successful_generations < num_samples:
            completion = self.pipe(prompt)[0]['generated_text']

            if not self.is_completion:
                completion = completion.replace(prompt, '', 1).strip()
            
            results.append(completion)
            successful_generations += 1
            
        return results
            
        
