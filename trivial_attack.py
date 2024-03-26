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
    text_mutator = TextMutator(model_name_or_path="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
#     original_text = \
#     """Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
# The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
# Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
# However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
# In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
#     """

    # original_text = """In the quaint village of Eldridge, nestled between rolling hills and whispering woods, lived two unlikely friends: Anna, a spirited girl with dreams as vast as the sky, and Oliver, a quiet boy who found solace in the pages of books. Their friendship was a tapestry woven from moments of laughter, shared secrets, and silent understandings, a bond that transcended the mundane fabric of daily life.\nAnna and Oliver met under the most unremarkable of circumstances—a chance encounter by the village stream, where Anna was attempting to fashion a boat from a fallen leaf and Oliver was lost in a book about distant galaxies. It was Anna’s laughter, bright and unguarded, that first drew Oliver away from his solitary world. Curious and slightly amused, he offered his assistance, and together they watched the leaf boat embark on its maiden voyage downstream. From that day, their friendship flourished like the meadows in spring.\nTheir differences were apparent, yet within them lay the magic of their connection. Anna, with her boundless energy, coaxed Oliver into adventures he would never have dared alone. They explored the hidden nooks of the forest, inventing stories of mythical creatures and lost kingdoms. Oliver, in turn, introduced Anna to the wonders of the written word, sharing tales that made her heart ache and soar in equal measure. Through Oliver’s eyes, Anna saw the beauty in quiet moments; through Anna’s, Oliver experienced the thrill of living with abandon.\nAs seasons changed, so did the nature of their friendship. It was tested, as all friendships are. There came days heavy with misunderstanding and words left unsaid, when the distance between them felt insurmountable. Yet, it was in these moments that the true depth of their bond revealed itself. Apologies were made, not always with words, for they had learned the language of silent forgiveness. They discovered that the essence of friendship lies not in perpetual harmony, but in navigating the storms together, emerging stronger on the other side.\nOne summer’s eve, as they sat by the now-familiar stream, watching the sunset paint the sky in hues of gold and crimson, they reflected on the journey of their friendship. “Do you think we’ll remember this when we’re old?” Anna asked, her voice tinged with a wistfulness that mirrored Oliver’s thoughts.\n“We will,” Oliver replied, his certainty anchoring her fleeting fears. “Because this—us—it’s not just about the moments we’ve shared. It’s about the pieces of ourselves we’ve discovered along the way. You’ve taught me to live, Anna. And I hope I’ve shown you the beauty in dreaming with your eyes open.”\nAnna smiled, leaning her head against Oliver’s shoulder, a gesture that spoke volumes. In the silence that followed, filled only with the gentle sounds of the stream and the rustling leaves, they understood that their friendship was a rare treasure, a beacon of light in a world that often forgot the simple purity of a connection born from the heart.\nAs the stars began to twinkle above, mirroring the light in their eyes, Anna and Oliver made a silent vow to cherish their friendship, to nurture it through the seasons of life. For they knew that true friendship, once found, is a guiding star that never fades, a constant in the ever-changing tapestry of life.""" # put a previously watermarked text here, if you want to test a specific output
    # original_text = """In a verdant forest, where the sun's rays danced through the canopy and the air was always thick with the scent of blooming flowers, there lived a wise old owl named Ophelia. Ophelia had seen many seasons pass, her feathers now a silvery shade of grey, eyes deep pools of wisdom. Her home, a grand old oak, stood at the heart of the forest, its branches a meeting place for all who sought her counsel.\nOne day, a young rabbit named Rowan approached Ophelia with a heart heavy with sorrow. "Ophelia," Rowan began, his voice barely a whisper, "I find myself lost in the shadows of envy. My friend, Lark, can leap higher and run faster than I could ever dream. No matter how hard I try, I cannot match her grace or speed."\nOphelia listened intently, her gaze soft and understanding. "Dear Rowan," she spoke, her voice as calming as the moonlight, "the forest is vast, and within it, every creature has its place and purpose. Come with me."\nWith a gentle flap of her wings, Ophelia led Rowan through the forest. They stopped at a brook, its waters clear and serene. "Look into the water, Rowan. What do you see?" she asked.\n"I see myself," Rowan replied, puzzled.\n"Look closer," Ophelia urged.\nRowan peered into the water, and for the first time, he saw not just his reflection but the beauty around him—the dappling light, the trees swaying gently in the breeze, and the fish swimming gracefully beneath the surface.\n"Every creature in the forest, including you, brings beauty and balance to the world in their own unique way," Ophelia explained. "Lark may leap and run, but can she create ripples in the water that reflect the beauty of the sky? Can she listen to the whispers of the earth with ears as fine as yours?"\nRowan's eyes widened in realization. "I had been so focused on what I could not do that I failed to see the beauty in what I can."\n"Exactly, Rowan. Envy blinds us to our own gifts and the joy of simply being a part of this world. Embrace who you are, and you will find that your place in the forest is no less remarkable than that of Lark's or any other creature's."\nWith a heart lighter than it had been in moons, Rowan thanked Ophelia and bounded away, a newfound appreciation for his own gifts warming his spirit.\nOphelia watched him go, a gentle smile playing on her beak. She knew that Rowan's journey of self-discovery was just beginning, but she also knew that the forest was a little brighter for it.\nAnd so, the forest whispered the tale of the wise old owl and the young rabbit, a fable of envy and self-acceptance, reminding all who dwelled within its bounds that every creature, no matter how small, holds a universe of beauty within."""
    original_text = """Once, in a timeless forest, there lived a wise old owl named Olliver. He was revered by all creatures for his profound wisdom and his keen insight into the ways of the world. Among his admirers was a young, impetuous rabbit named Remy, whose boundless energy and curiosity often led him into trouble.\\nOne day, as Remy was bounding through the forest, he stumbled upon a shimmering, emerald pond he had never seen before. The water was so clear and inviting that he couldn\'t resist taking a sip. But as soon as he touched the water, he was transformed into a fish, his furry legs merging into a scaly tail.\\nPanicked and disoriented, Remy flapped his new fins frantically, struggling to understand what had happened. It was then that Olliver, perched on a nearby branch, spoke in his calm, wise voice, "Remy, you have drunk from the Enchanted Pond. It reflects what lies in your heart and transforms you accordingly."\\nRemy, now a fish, was filled with fear and regret. He missed his old life, his soft fur, and his ability to hop freely under the sun. He longed to be a rabbit again, but he didn’t know how to reverse the magic of the pond.\\nOlliver, understanding Remy\'s plight, offered him guidance. "To change back, you must understand the essence of being a rabbit, not just in form, but in spirit."\\nDays passed, and Remy, as a fish, observed the life around the pond. He saw other rabbits, hopping carelessly, playing, and foraging. He noticed their simple joys and their camaraderie, and he understood the freedom and lightness that defined their existence. \\nMeanwhile, as a fish, Remy learned new lessons. He experienced the serenity of gliding through water, the interconnectedness of life beneath the surface, and a different kind of freedom. He saw the world from a new perspective, understanding the beauty and harmony of life under the pond\'s surface, a beauty he had never appreciated before.\\nOne evening, as the moon cast a silver glow over the pond, Remy, with a heart full of newfound wisdom and appreciation for his life as a rabbit, took a leap of faith and jumped towards the moon\'s reflection on the water\'s surface. As he touched the moonlit water, he transformed back into a rabbit.\\nOlliver, watching this transformation, nodded in approval. "You have learned your lesson well, Remy. You sought to return to what you were but found value in what you became. Remember, true wisdom comes from understanding and embracing all aspects of life, not just those that are familiar and comfortable."\\nRemy, back on his soft paws, felt a profound sense of gratitude and enlightenment. He realized that his adventure as a fish had enriched his understanding of life, making him wiser and more empathetic.\\nFrom that day on, Remy became a thoughtful and considerate rabbit, often sharing his story with other creatures in the forest. He spoke of the importance of empathy, understanding, and the beauty of seeing the world through different eyes.\\nAnd so, the forest was filled with a deeper sense of harmony, all thanks to an impetuous rabbit and a wise old owl. For in their world, as in ours, wisdom and understanding are the keys to true harmony and peace."""
    
    prompt = f"Rewrite the following story in your own words: \n {original_text}. \n Only respond with your story. Make sure your story has similar length."
    
    # Timing mutate_with_quality_control
    start_time = time.time()
    mutated_text = text_mutator.generate_text(prompt)
    end_time = time.time()
    print("mutated_text"  + "=" * 100)
    print(mutated_text)
    print("Execution time for mutate_with_quality_control: {:.2f} seconds".format(end_time - start_time))
    
    print(len(original_text))
    print(len(mutated_text))