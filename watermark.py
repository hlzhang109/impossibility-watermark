import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

import logging
import torch
import numpy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from pipeline_builder import PipeLineBuilder
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

log = logging.getLogger(__name__)

class Watermarker:
    def __init__(self, cfg, pipeline=None, n_attempts=5):
        self.cfg = cfg # config.watermark_args
        self.n_attempts = n_attempts
        self.pipeline = pipeline
        
        if not isinstance(self.pipeline, PipeLineBuilder):
            log.info("Initializing a new Watermarker pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg.generator_args)

        # Extract Model and Tokenizer from Piepline to manipulate token probs
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

         # Define generation config
        self.generator_kwargs = {
            "max_new_tokens": self.cfg.generator_args.max_new_tokens,
            "do_sample": self.cfg.generator_args.do_sample,
            "temperature": self.cfg.generator_args.temperature,
            "top_p": self.cfg.generator_args.top_p,
            "top_k": self.cfg.generator_args.top_k,
            "repetition_penalty": self.cfg.generator_args.repetition_penalty
        }

        # Initialize the watermark processor and detector objects
        if self.cfg.watermark_args.name == "umd":
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
            
        elif self.cfg.watermark_args.name == "unigram":
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
           
        # Additional parameters based on watermarking scheme
        if self.cfg.watermark_args.name == "umd":
            self.generator_kwargs["logits_processor"] = LogitsProcessorList([self.watermark_processor])
        elif self.cfg.watermark_args.name == "unigram":
            self.generator_kwargs["logits_processor"] = self.watermark_processor
            self.generator_kwargs["output_scores"] = True
        elif self.cfg.watermark_args.name == "exp":
            torch.manual_seed(self.cfg.watermark_args.seed)
            self.watermark_sequence_length = self.cfg.watermark_args.watermark_sequence_length
            self.generated_text_length = self.cfg.watermark_args.generated_text_length
            self.watermark_sequence_key = self.cfg.watermark_args.watermark_sequence_key
            self.p_threshold = self.cfg.watermark_args.p_threshold
            
    def generate(self, prompt):
        n_attempts = 0
        while n_attempts < self.n_attempts:

            # Attempt to generate response with watermark
            if self.cfg.watermark_args.name == "exp":
                tokens = self.tokenizer.encode(
                    prompt, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=self.cfg.generator_args.max_new_tokens)
                outputs = generate_shift(
                    model=self.model,
                    prompt=tokens,
                    vocab_size=len(self.tokenizer),
                    n=self.watermark_sequence_length,
                    m=self.generated_text_length,
                    key=self.watermark_sequence_key
                )
            else:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    # padding=True, 
                    truncation=True, 
                    max_length=self.cfg.generator_args.max_new_tokens
                ).to(self.model.device)
                outputs = self.model.generate(**inputs, **self.generator_kwargs)
                
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = completion.replace(prompt, '', 1).strip()

            # Check if watermark succeeded
            is_detected, _ = self.detect(completion)
            if is_detected:
                return completion
            else:
                log.info("Failed to watermark, trying again...")
                n_attempts += 1

        return None
        
    def detect(self, completion):
        if self.cfg.watermark_args.name == "umd":  
            score = self.watermark_detector.detect(completion)
            score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
            z_score = score_dict['z_score']
            is_detected = score_dict['prediction']
            return is_detected, z_score
        elif self.cfg.watermark_args.name == "unigram":
            token_sequence = self.tokenizer(completion, add_special_tokens=False)['input_ids']
            z_score = self.watermark_detector.detect(token_sequence, device=self.model.device)
            is_detected = (z_score >= self.cfg.watermark_args.z_threshold)
            return is_detected, z_score
        elif self.cfg.watermark_args.name == "exp":
            tokens = self.tokenizer.encode(completion, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
            pval = permutation_test(tokens,self.watermark_sequence_key,self.watermark_sequence_length,len(tokens),len(self.tokenizer))
            is_detected = (pval <= self.p_threshold)
            return is_detected, pval
        return None
        

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):
    import time
    import textwrap
    
    prompt = textwrap.dedent(
        """Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings 
        series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?"""
    )

    watermarker = Watermarker(cfg)

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

# [2024-01-22 21:59:36,673][__main__][INFO] - Watermarked Text: The role of power in J.R.R Tolkien's epic fantasy novel, The Lord of the Rings, is prominent throughout each book. Power can be defined as a person's ability to control others or influence events. The One Ring that serves as the central plot device symbolizes power in this trilogy. The One Ring holds the ability to govern every other ring, which has been forged by the Dark Lord Sauron himself. It offers an opportunity to become all-powerful, but with such power comes corruption, destruction, manipulation, domination, and evil. The One Ring symbolizes power because it grants the wielder authority over every aspect of Middle-Earth. Still, at the same time, it takes away free will and humanity from those who hold it. The characters who interact with the One Ring are affected differently based on their inner natures. The hobbit Frodo Baggins shows resistance towards its call initially, whereas his best friend Samwise Gamgee tries hard not to fall under its sway. In contrast, Gollum becomes consumed by the ring’s power until he turns into something monstrous. However, even without the One Ring's influence, some characters exhibit their hunger for power through politics or warfare like Saruman or Denethor. Throughout the series, Tolkien suggests that power corrupts individuals when they abuse or misuse it. He also emphasizes that too much concentrated power can lead only to tyranny and devastation. The author uses various character arcs throughout the novels to illustrate these points subtly while maintaining an engaging narrative about friendship, loyalty, bravery, and sacrifice. Overall, J.R.R Tolkien suggests that true power lies not in controlling others but rather empowering oneself with self-awareness, compassion, empathy, courage, resilience, and humility.
# [2024-01-22 21:59:36,674][__main__][INFO] - Is Watermark Detected?: True
# [2024-01-22 21:59:36,674][__main__][INFO] - Score: 8.725672342591785
# [2024-01-22 21:59:36,674][__main__][INFO] - Time taken: 63.595707654953