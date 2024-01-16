import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class GenerativeModel:
    def __init__(self, model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", 
                 revision="main", # "gptq-3bit-128g-actorder_True" for 3-bit version
                 max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95, 
                 top_k=40, repetition_penalty=1.1, cache_dir="./.cache/"):

        # Initialize and load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map="cuda",
            trust_remote_code=False,
            revision=revision) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, cache_dir=cache_dir)

        # Store the pipeline configuration
        self.pipeline_config = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }

        # Create the pipeline
        self.pipe = pipeline("text-generation", **self.pipeline_config)
        
        # Check if NLTK data is downloaded, if not, download it
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    
    def generate_text(self, prompt):
        """
        This function expects a formatted prompt and returns the generated text.
        """
        return self.pipe(prompt)[0]['generated_text'].replace(prompt, "").strip()
        
    def __call__(self, prompt):
        return self.generate_text(prompt)
