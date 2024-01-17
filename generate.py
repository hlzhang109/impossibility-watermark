from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, AutoConfig

# UMD
from extended_watermark_processor import WatermarkLogitsProcessor
# Unigram
from gptwm import GPTWatermarkLogitsWarper

class TextGenerator:
    # TODO: Add C4 dataset support.
    # TODO: Add EXP scheme.
    def __init__(self, args):
        self.watermarking_scheme = args.watermarking_scheme
        
        self.model_name = args.model_name
        
        print(f"Text Generation Model: {self.model_name}")
        print(f"Watermarking Scheme: {self.watermarking_scheme}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.config = AutoConfig.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                            device_map="auto",
                                            trust_remote_code=False,
                                            revision="main",
                                            config=self.config)
        # TODO: Add completion vs. prompt generation.
        self.is_completion = False
    
        if self.watermarking_scheme == "umd": 
            self.watermark_processor = WatermarkLogitsProcessor(vocab=list(self.tokenizer.get_vocab().values()),
                                                        gamma=0.25,
                                                        delta=2.0,
                                                        seeding_scheme="selfhash")
        elif self.watermarking_scheme == "unigram":
            # TODO: Make the arguments to Unigram more systematic.
            wm_key = 0
            self.watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=0.5,
                                                                        strength=2.0,
                                                                        vocab_size=self.tokenizer.vocab_size,
                                                                        watermark_key=wm_key)])

    def generate(self, prompt, num_samples, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1024):
        results = []
        gen_kwargs = {
            "temperature": temperature, 
            "do_sample": do_sample, 
            "top_p": top_p, 
            "top_k": top_k, 
            "max_new_tokens": max_new_tokens
        }
        
        # Additional parameters based on watermarking scheme
        if self.watermarking_scheme == "umd":
            gen_kwargs["logits_processor"] = LogitsProcessorList([self.watermark_processor])
        elif self.watermarking_scheme == "unigram":
            gen_kwargs["logits_processor"] = self.watermark_processor
            gen_kwargs["output_scores"] = True
            
        successful_generations = 0
        while successful_generations < num_samples:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100).to(self.model.device)
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            print(len(outputs))
            
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if not self.is_completion:
                completion = completion.replace(prompt, '', 1)
            
            results.append(completion)
            successful_generations += 1
            
        return results
            
        
