from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, T5ForConditionalGeneration, GPTQConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from utils import mixtral_format_instructions, parse_llama_output

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import logging
import hydra

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class PipeLineBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.requires_INST_tokens = False
        
        log.info(f"Initializing {cfg.model_name_or_path}")

        # NOTE: Using openai is incompatible with watermarking. 
        if "gpt" in cfg.model_name_or_path:
            load_dotenv(find_dotenv()) # load openai api key from ./.env
            self.pipeline = ChatOpenAI(model_name=cfg.model_name_or_path)
        
        else: # Using locally hosted model
            self._init_model(self.cfg)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name_or_path, 
                use_fast=True, 
                cache_dir=cfg.model_cache_dir)

            self._init_pipeline_config(self.cfg)

            # Create the pipeline
            if "grammarly" in cfg.model_name_or_path:
                self.pipeline_base = pipeline("text2text-generation", **self.pipeline_config)
                self.pipeline = HuggingFacePipeline(pipeline=self.pipeline_base)                 
            else:
                self.pipeline_base = pipeline("text-generation", **self.pipeline_config)
                self.pipeline = HuggingFacePipeline(pipeline=self.pipeline_base) 

    def _init_model(self, cfg):
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
            
        elif "Mixtral" in cfg.model_name_or_path:            
            # Initialize and load the model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                revision=cfg.revision,
                cache_dir=cfg.model_cache_dir,
                device_map=cfg.device_map,
                trust_remote_code=cfg.trust_remote_code
                )
            
        # NOTE: We might want to remove this if we don't end up using it.
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
    
    def _init_pipeline_config(self,cfg):
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

        if 'Llama' in cfg.model_name_or_path:
            stop_token = 'assistant'
            stop_token_id = self.tokenizer.encode(stop_token, return_tensors='pt')
            self.pipeline_config['eos_token_id'] = [self.tokenizer.eos_token_id, stop_token_id]
        
        if 'Mixtral' in cfg.model_name_or_path:
            self.pipeline_config["return_full_text"] = False
            self.requires_INST_tokens = True

    def generate_text(self, prompt: PromptTemplate):
        """
        This function expects a prompt and returns the generated text.
        """
        prompt_str = str(prompt)
        log.info(f"Prompt: {prompt_str}")

        if "gpt" in self.cfg.model_name_or_path:
            return self.pipeline(prompt)
        
        if "Llama" in self.cfg.model_name_or_path:
            messages = [
                # {"role": "system", "content": "You are a helpful personal assistant."}, # NOTE: Removed this since the responses always started with "As a helpful personal assistant," - Boran
                {"role": "user", "content": prompt},
            ]

            generated_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            log.info(f"Generated Prompt for Llama: {generated_prompt}")

            output = self.pipeline(generated_prompt) 
            prompt_end_index = output.find(prompt_str) + len(prompt_str)
            if prompt_end_index != -1:
                output = output[prompt_end_index:].strip()

            return parse_llama_output(output)

        if "Mixtral" in self.cfg.model_name_or_path:
            prompt = mixtral_format_instructions(prompt)

        if not isinstance(prompt, str):
            prompt = prompt.to_string()
        return self.pipeline(prompt)

    def __call__(self, prompt: PromptTemplate):
        return self.generate_text(prompt)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Create or get existing pipeline builders for generator, oracle, and mutator.
    mutator_pipeline_builder = PipeLineBuilder(cfg.mutator_args)
    prompt = "Describe the main responsibilities of a U.S. Senator."
    response = mutator_pipeline_builder(prompt)
    print(response)

if __name__ == "__main__":
    main()