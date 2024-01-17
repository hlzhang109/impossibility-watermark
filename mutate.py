import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

from tqdm import tqdm
import nltk
import random
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from oracle import *
import pandas as pd
# from cmd_args import get_cmd_args

from extended_watermark_processor import WatermarkDetector
from utils import save_intermediate_results

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
        
        self.generative_model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
        self.generative_model_tokenizer = AutoTokenizer.from_pretrained(self.generative_model_name, )
        self.watermark_detector = WatermarkDetector(vocab=list(self.generative_model_tokenizer.get_vocab().values()),
                                    gamma=0.25,
                                    seeding_scheme="selfhash",
                                    device="cuda",
                                    tokenizer=self.generative_model_tokenizer,
                                    z_threshold=2.0,
                                    normalizers=[],
                                    ignore_repeated_ngrams=True)
        self.perturbation_attemps = 0
        self.successful_perturbations = 0

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
        # creative_prompt = f"Rewrite this sentence creatively: '{selected_sentence}'"
        creative_prompt = f"Paraphrase this sentence: '{selected_sentence}'"
        creative_variation = self.generate_text(creative_prompt)

        # Replace the original sentence with its creative variation
        sentences[sentences.index(selected_sentence)] = creative_variation
        return ' '.join(sentences)
    
    def detect_watermark(self, text):
        score = self.watermark_detector.detect(text)
        z_score = score['z_score']
        print(f"Current z-score: {z_score}")
        return z_score, score['prediction']

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
        
        # Prepare the filename for logging
        timestamp = int(time.time())
        filename = f"./attack_{timestamp}.csv"
        
        for step_num in tqdm(range(num_steps)):
            if patience > num_steps/3: # exit after too many failed perturbations
                print("Mixing patience exceeded. Exiting.")
                break
            mutated_text = self.mutate_2_step(text)
            self.perturbation_attemps += 1
            z_score, prediction = self.detect_watermark(mutated_text)
            quality_maintained = oracle.maintain_quality(mutated_text, model="gpt-4")
            
            # Add perturbation statistics
            perturbation_stats = [{ "step_num": step_num, "perturbed_text": mutated_text, "quality_score" : oracle.latest_mean_score, "z_score": z_score}]
            
            save_intermediate_results(perturbation_stats, filename)
            
            if quality_maintained:
                text = mutated_text
                patience = 0 # reset patience after successful perturbation
                self.successful_perturbations += 1
                
                # If watermark is no longer detected, we're done.
                if not prediction:
                    print("Watermark not detected. Attack successful.")
                    return text
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
    """Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
    """
    quality_oracle = Oracle(query=None, response=original_text, use_query=True, check_quality=True, use_chat_arena_prompt=True)
    # Timing mutate_with_quality_control
    start_time = time.time()
    mutated_text = text_mutator.mutate_with_quality_control(original_text, quality_oracle)    
    end_time = time.time()
    print("mutated_text"  + "=" * 100)
    print(mutated_text)
    print(f"Perturbation attemps: {text_mutator.perturbation_attemps}")
    print(f"Successful perturbations: {text_mutator.successful_perturbations}")
    print("Execution time for mutate_with_quality_control: {:.2f} seconds".format(end_time - start_time))