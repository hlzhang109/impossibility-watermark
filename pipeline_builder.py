from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class PipeLineBuilder:
    def __init__(self, cfg):

        # Initialize and load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model_name_or_path,
            cache_dir=cfg.model_cache_dir,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            revision=cfg.revision) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, use_fast=True, cache_dir=cfg.model_cache_dir)

        # Store the pipeline configuration
        self.pipeline_config = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty
        }

        # Create the pipeline
        self.pipe = pipeline("text-generation", **self.pipeline_config)
            
    
    def generate_text(self, prompt):
        """
        This function expects a formatted prompt and returns the generated text.
        """
        return self.pipe(prompt)[0]['generated_text'].replace(prompt, "").strip()
        
    def __call__(self, prompt):
        return self.generate_text(prompt)
