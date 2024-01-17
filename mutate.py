import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

import logging
import textwrap
from tqdm import tqdm
import nltk
import random
from nltk.tokenize import sent_tokenize
from pipeline_builder import PipeLineBuilder
from oracle import Oracle

from utils import save_to_csv

log = logging.getLogger(__name__)

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
    def __init__(self, cfg, watermarker, quality_oracle):

        self.cfg = cfg
        self.pipe = PipeLineBuilder(cfg)
        self.watermarker = watermarker
        self.quality_oracle = quality_oracle

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
        log.info(f"Current z-score: {z_score}")
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
    
    def mutate_with_quality_control(self, text):
        """
        Mutate the text for a given number of steps with quality control.
        """
        perturbation_attemps = 0
        successful_perturbations = 0

        patience = 0
        
        # Prepare the filename for logging
        save_path = self.cfg.save_name.replace("{time_stamp}", int(time.time()))
        
        initial_state = [{
            "step_num": -1, 
            "text": text, 
            "quality_score" : -1, 
            "quality_maintained": quality_maintained,
            "z_score": -1,
            "watermark_detected": -1
        }]
        
        save_to_csv(initial_state, save_path)
        
        for step_num in tqdm(range(self.cfg.num_steps)):

            if patience > self.cfg.patience: # exit after too many failed perturbations
                log.info("Mixing patience exceeded. Exiting.")
                break

            mutated_text = self.mutate_2_step(text)
            perturbation_attemps += 1
            z_score, watermark_detected = self.detect_watermark(mutated_text)
            quality_maintained = self.quality_oracle.maintain_quality(mutated_text, model="gpt-4")
            
            # Add perturbation statistics
            perturbation_stats = [{
                "step_num": step_num, 
                "text": mutated_text, 
                "quality_score" : self.quality_oracle.latest_mean_score, 
                "quality_maintained": quality_maintained,
                "z_score": z_score,
                "watermark_detected": watermark_detected
            }]
            
            save_to_csv(perturbation_stats, save_path)
            
            if quality_maintained:
                text = mutated_text
                patience = 0 # reset patience after successful perturbation
                successful_perturbations += 1
                log.info("SUCCESSFUL PERTURBATION!")
                
                # If watermark is no longer detected, we're done.
                if not watermark_detected:
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
    original_text = textwrap.dedent("""
        Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
        The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
        Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
        However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
        In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
    """)
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