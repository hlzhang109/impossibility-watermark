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

from extended_watermark_processor import WatermarkDetector

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
                                    z_threshold=4.0,
                                    normalizers=[],
                                    ignore_repeated_ngrams=True)

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
        print(f"Current z-score: {score['z_score']}")
        return score['prediction']

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
                
                # If watermark is no longer detected, we're done.
                if not self.detect_watermark(text):
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
    """
    In J.R.R. Tolkien's The Lord of the Rings, power is a central theme that shapes the actions and motivations of many characters. The most significant example of power in the series is the One Ring, which was created by Sauron to bring about his rule over Middle-earth. The Ring represents the ultimate form of power, as it allows its owner to control and govern the entire world. However, the power of the Ring also comes with a terrible cost, as it corrupts and enslaves those who possess it.\nTolkien suggests that power can be both seductive and dangerous. Those who seek power too eagerly or too frequently become corrupted by it, such as Boromir and Gollum. On the other hand, characters like Aragorn and Samwise Gamgee are able to resist the temptation of power and remain pure of heart. This highlights the idea that true power is not about having authority or control but rather about using one's abilities for good.\nThe struggle between good and evil is another important aspect of the book's exploration of power. Sauron's desire for dominance over Middle-earth represents the darker side of power, while the Fellowship's efforts to destroy the Ring embody the aspirations of good. Through this conflict, Tolkien shows how power can be used to create either order or chaos, depending on the intentions of the individual wielding it.\nOverall, the Lord of the Rings illustrates how power can be both alluring and harmful, and how it may be utilized for good or bad. Tolkien uses the One Ring as a symbol of power, illuminating its allure and dangers. He also demonstrates how people might make decisions based on their moral compass and how power might be used for good or evil. Power significantly influences the characters' actions in the Lord of the Rings, and it is a topic well worth investigating.
    """
    quality_oracle = Oracle(query=None, response=original_text, use_query=True, check_quality=True, use_chat_arena_prompt=True)
    # Timing mutate_with_quality_control
    start_time = time.time()
    mutated_text = text_mutator.mutate_with_quality_control(original_text, quality_oracle)
    end_time = time.time()
    print("mutated_text"  + "=" * 100)
    print(mutated_text)
    print("Execution time for mutate_with_quality_control: {:.2f} seconds".format(end_time - start_time))