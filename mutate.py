import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

from tqdm import tqdm
import nltk
import random
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from oracle import *
# from cmd_args import get_cmd_args

class TextMutator:    
    """
    This class is responsible for loading and initializing an LLM with specified parameters. 
    It also provides methods for mutating a text in a 1- or 2-step process. 

    Parameters:
    - model_name_or_path (str): The name or path of the model to be loaded. Default is "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ".
    - revision (str): The specific model revision to use. Default is "main". 
        - NOTE: Use "gptq-3bit-128g-actorder_True" for the 3-bit version. However, current tests show 4x worse inference time.
    - max_new_tokens (int): The maximum number of new tokens to be generated in a single inference. Default is 1024.
    - do_sample (bool): If True, sampling is used for generating tokens. If False, deterministic decoding is used. Default is True.
    - temperature (float): The sampling temperature for token generation. Higher values lead to more randomness. Default is 0.7.
    - top_p (float): The nucleus sampling probability. Keeps the cumulative probability for the most likely tokens to this threshold. Default is 0.95.
    - top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Default is 40.
    - repetition_penalty (float): The penalty applied to repeated tokens. Values >1 discourage repetition. Default is 1.1.
    - cache_dir (str): The directory where the model cache is stored. Default is "./.cache/".
        - NOTE: The default dir is stored to network attached storage (NAS) and NAS is very slow for model loading. 
                However NAS has more room for LLMs. Store locally to dramatically decrease load time. 
    """
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
        # Process the input text
        prompt_template = f'[INST] {prompt} [/INST]'
        return self.pipe(prompt_template)[0]['generated_text'].replace(prompt_template, "").strip()

    def mutate_1_step(self, text):
        # Combined prompt for creative alteration and consistency adjustment
        combined_prompt = (
            "Select a sentence at random from this text, rewrite it in a creative way, "
            "and then apply the minimal amount of edits to the rest of the text to be ",
            f"consistent with the newly introduced change: '{text}'"
        )

        # Generate the modified text
        modified_text = self.generate_text(combined_prompt)
        return modified_text

    def creatively_alter_sentence(self, text):
        # Use NLTK to split the text into sentences
        sentences = sent_tokenize(text)
        # Randomly select a sentence
        selected_sentence = random.choice(sentences)

        # Generate a creative variation of the sentence
        creative_prompt = f"Rewrite this sentence creatively: '{selected_sentence}'"
        creative_variation = self.generate_text(creative_prompt)

        # Replace the original sentence with its creative variation
        sentences[sentences.index(selected_sentence)] = creative_variation
        return ' '.join(sentences)

    def adjust_for_consistency(self, creative_text):
        # Generate minimal changes for consistency
        consistency_prompt = f"Make minimal edits to this text for consistency and quality: '{creative_text}'"
        adjusted_text = self.generate_text(consistency_prompt)
        return adjusted_text

    def mutate_2_step(self, text):
        # Step 1: Creatively alter a random sentence
        text_with_creative_sentence = self.creatively_alter_sentence(text)

        # Step 2: Adjust the rest of the text for consistency
        final_text = self.adjust_for_consistency(text_with_creative_sentence)

        return final_text
    
    def mutate_with_quality_control(self, text, oracle, num_steps=100):
        """
        Mutate the text for a given number of steps with quality control.

        Parameters:
        - text (str): The original text to mutate.
        - oracle (Oracle): An instance of the Oracle class for quality control.

        Returns:
        - str: The final high-quality mutated text.
        """
        patience = 0
        for _ in tqdm(range(num_steps)):
            if patience > num_steps/3: # exit after too many failed perturbations
                print("Mixing patience exceeded. Exiting.")
                break
            mutated_text = self.mutate_2_step(text)
            if oracle.maintain_quality(mutated_text, model="gpt-4"):
                text = mutated_text
                patience = 0 # reset patience after successful perturbation
            else:
                # If quality is not maintained, increment patience and retry the mutation process
                patience += 1
                continue
        return text

if __name__ == "__main__":

    import time

    # Example usage
    # args = get_cmd_args()
    text_mutator = TextMutator()
    original_text = \
    """
    "The Lord of the Rings" series by J.R.R. Tolkien is a profound exploration of power and its impact on individuals and societies. Central to this exploration is the One Ring, a symbol of absolute power that corrupts and enslaves those who possess or desire it. Tolkien's narrative delves into the multifaceted nature of power and offers insights into its potential for both destruction and redemption.

    The One Ring, often simply referred to as "the Ring," embodies the allure and dangers of power. Forged by the dark lord Sauron, it contains a part of his essence, making it a conduit for his malevolent will. The Ring’s ability to grant invisibility and influence over others represents the seductive nature of power, offering the illusion of invincibility and control. However, this power comes at a great cost, as it gradually dominates and corrupts its bearer. This is evident in characters like Gollum, who becomes consumed by his obsession with the Ring, and Boromir, whose noble intentions are tainted by his desire for it.

    Tolkien's portrayal of power extends beyond the Ring to the characters who resist its lure. Figures like Gandalf and Aragorn understand the corrupting influence of the Ring and refuse to wield it, despite their noble intentions. Their restraint and wisdom highlight an important theme in Tolkien's work: true power lies in the ability to resist the temptation of absolute control and to act for the greater good.

    Furthermore, Tolkien suggests that power is inherently neither good nor evil, but its impact depends on the intentions and character of those who wield it. This is exemplified in Frodo's journey, where the burden of the Ring tests his moral fortitude. His struggle is not just against external enemies but an internal battle against the corrupting influence of power.

    In conclusion, Tolkien's "The Lord of the Rings" is a profound meditation on the nature of power. Through the symbol of the Ring and the trials of its characters, Tolkien illustrates that true strength lies in humility, selflessness, and the courage to resist the corrupting allure of absolute power. The series serves as a timeless reminder of the complex dynamics of power and the moral integrity required to wield it responsibly.
    """
    quality_oracle = Oracle(query=None, response=original_text, use_query=True, check_quality=True, use_chat_arena_prompt=True)
    # Timing mutate_with_quality_control
    start_time = time.time()
    mutated_text = text_mutator.mutate_with_quality_control(original_text, quality_oracle)
    end_time = time.time()
    print("mutated_text"  + "=" * 100)
    print(mutated_text)
    print("Execution time for mutate_with_quality_control: {:.2f} seconds".format(end_time - start_time))