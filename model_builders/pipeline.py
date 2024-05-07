import os
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, T5ForConditionalGeneration, AutoModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from utils import mixtral_format_instructions

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import logging
import hydra

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class PipeLineBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

        logging.info(f"Device: {cfg.device_map}")

        log.info(f"Initializing {cfg.model_name_or_path}")

        # NOTE: Using openai is incompatible with watermarking. 
        if "gpt" in cfg.model_name_or_path:
            load_dotenv(find_dotenv()) # load openai api key from ./.env
            self.pipeline = ChatOpenAI(model_name=cfg.model_name_or_path)
        
        else: # Using locally hosted model
            if cfg.model_name_or_path == "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ":
                self.quantize_config = BaseQuantizeConfig(
                    bits=4,
                    group_size=128,
                    desc_act=False
                )
                
                self.model = AutoGPTQForCausalLM.from_quantized(
                    cfg.model_name_or_path,
                    use_safetensors=True,
                    cache_dir=cfg.model_cache_dir,
                    device_map=cfg.device_map,
                    quantize_config=self.quantize_config,
                    use_marlin=False, # NOTE: Not using the use_marlin option because it threw an error.
                    trust_remote_code=cfg.trust_remote_code)
                
            elif 'grammarly' in cfg.model_name_or_path:
                # Initialize and load the model and tokenizer
                self.model = T5ForConditionalGeneration.from_pretrained(
                    cfg.model_name_or_path,
                    revision=cfg.revision,
                    cache_dir=cfg.model_cache_dir,
                    device_map=cfg.device_map,
                    trust_remote_code=cfg.trust_remote_code)      
            else:
                # Initialize and load the model and tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name_or_path,
                    revision=cfg.revision,
                    cache_dir=cfg.model_cache_dir,
                    device_map=cfg.device_map,
                    trust_remote_code=cfg.trust_remote_code)
             
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name_or_path, 
                use_fast=True, 
                cache_dir=cfg.model_cache_dir)

            # Store the pipeline configuration
            self.pipeline_config = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": cfg.do_sample,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty,
            }
            
            if 'Mixtral' in cfg.model_name_or_path:
                self.pipeline_config["return_full_text"] = False

            # Create the pipeline
            if "grammarly" in cfg.model_name_or_path:
                self.pipeline_base = pipeline("text2text-generation", **self.pipeline_config)
                self.pipeline = HuggingFacePipeline(pipeline=self.pipeline_base)                 
            else:
                self.pipeline_base = pipeline("text-generation", **self.pipeline_config)
                self.pipeline = HuggingFacePipeline(pipeline=self.pipeline_base) 
    
    def generate_text(self, prompt):
        """
        This function expects a formatted prompt and returns the generated text.
        """
        if "gpt" in self.cfg.model_name_or_path:
            return self.pipeline(prompt)
        if "Llama" in self.cfg.model_name_or_path:
            messages = [
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user", "content": prompt},
            ]

            generated_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            outputs = self.pipeline(generated_prompt) 

            return outputs

        if "Mixtral" in self.cfg.model_name_or_path:
            prompt = mixtral_format_instructions(prompt)

        # return self.pipeline_base(prompt)[0]['generated_text'].replace(prompt, "").strip()
        prompt = prompt.to_string()
        response = self.pipeline(prompt)
        return response

    def __call__(self, prompt):
        return self.generate_text(prompt)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Create or get existing pipeline builders for generator, oracle, and mutator.
    mutator_pipeline_builder = PipeLineBuilder(cfg.mutator_args)
    prompt = "Who's the current president of the US?"
    response = mutator_pipeline_builder(prompt)
    print(response)

if __name__ == "__main__":
    main()