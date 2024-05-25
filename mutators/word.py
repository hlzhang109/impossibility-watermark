from transformers import pipeline
import random
import re
import difflib
import hydra
import logging

log = logging.getLogger(__name__)

class MaskFillMutator:
    def __init__(self, model_name="FacebookAI/roberta-large"):
        self.model_name = model_name
        self.max_length = 256
        # NOTE: Currently does not use GPU
        self.fill_mask = pipeline(
            "fill-mask", 
            model=self.model_name, 
            tokenizer=self.model_name
        )
        self.tokenizer_kwargs = {"truncation": True, "max_length": 512}

    def get_words(self, text):
        # Use a more comprehensive regex to capture more types of trailing punctuation
        m = re.match(r'^(.*?)([\.!?,:;()-]+)?$', text)
        core_text, end_punctuation = m.groups() if m else (text, None)
        words = core_text.split()
        return words, end_punctuation

    def select_random_segment(self, words):
        if len(words) <= self.max_length:
            return words, 0, len(words)
        start_index = random.randint(0, len(words) - self.max_length)
        return words[start_index:start_index + self.max_length], start_index, start_index + self.max_length

    def mask_random_word(self, words):
        if not words:  # Return the original text if there are no words to mask
            return words, None

        index_to_mask = random.randint(0, len(words) - 1)  # Select a random index to mask
        word_to_mask = words[index_to_mask]  # Get the word at the selected index

        # Create masked text by replacing only the selected word
        masked_text = ' '.join('<mask>' if i == index_to_mask else word for i, word in enumerate(words))
        return masked_text, word_to_mask

    def get_highest_score_index(self, suggested_replacements, blacklist):
        # Filter out dictionaries where 'token_str' is a blacklisted word
        filtered_data = [d for d in suggested_replacements if d['token_str'].strip().lower() not in blacklist]

        # Find the index of the dictionary with the highest score
        if filtered_data:
            highest_score_index = max(range(len(filtered_data)), key=lambda i: filtered_data[i]['score'])
            return filtered_data[highest_score_index]
        else:
            return suggested_replacements[0]

    def mutate(self, text, num_replacements=0.01):
        words, end_punctuation = self.get_words(text)

        if len(words) > self.max_length:
            segment, start, end = self.select_random_segment(words)
        else:
            segment, start, end = words, 0, len(words)

        if num_replacements < 0:
            raise ValueError("num_replacements must be larger than 0!")
        if 0 < num_replacements < 1:
            num_replacements = max(1, int(len(segment) * num_replacements))

        log.info(f"Making {num_replacements} replacements to the input text segment.")

        replacements_made = 0
        while replacements_made < num_replacements:
            masked_text, word_to_mask = self.mask_random_word(segment)
            candidates = self.fill_mask(masked_text, top_k=3, tokenizer_kwargs=self.tokenizer_kwargs)
            suggested_replacement = self.get_highest_score_index(candidates, blacklist=[word_to_mask.lower()])
            log.info(f"word_to_mask: {word_to_mask}")
            log.info(f"suggested_replacement: {suggested_replacement['token_str']} (score: {suggested_replacement['score']})")
            segment = suggested_replacement['sequence'].split()
            replacements_made += 1

        if end_punctuation:
            segment[-1] += end_punctuation

        combined_text = ' '.join(words[:start]) + ' ' + ' '.join(segment) + ' ' + ' '.join(words[end:])
        return self.cleanup(combined_text)

    def cleanup(self, text):
        return text.replace("<s>", "").replace("</s>", "")

    def diff(self, text1, text2):
        # Splitting the texts into lines as difflib works with lists of lines
        text1_lines = text1.splitlines()
        text2_lines = text2.splitlines()
        
        # Creating a Differ object
        d = difflib.Differ()

        # Calculating the difference
        diff = list(d.compare(text1_lines, text2_lines))

        # Joining the result into a single string for display
        diff_result = '\n'.join(diff)

        return diff_result

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test(cfg):

    import time
    import textwrap
    import os

    CUDA_VISIBLE_DEVICES = str(cfg.cuda_visible_devices)
    WORLD_SIZE = str(len(str(cfg.cuda_visible_devices).split(",")))

    print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}")
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["WORLD_SIZE"] = WORLD_SIZE

    text = """
    "Set in the 1990s, Captain Marvel follows Danvers, an Air Force officer, as she becomes one of the universe's most powerful heroes after earth gets involved in a galactic war between alien. There is no doubt that there are many superhero movies out there but it has been.
    Captain marvel and the kree-skrull war explained:
    Marvel Studios released the first trailer for Captain Marvel on September 18, 2018, and it gave us our first look at Brie Larson as Carol Danvers/Captain Marvel. However, the trailer didn't delve too deeply into the plot or the characters. But now, thanks to some new promo art and details from the film's stars and director, we have a better understanding of what's going on in the movie.

    Captain marvel trailer breakdown:

    The Captain Marvel trailer shows us a mix of action, humor, and emotional moments, giving us a glimpse into the life of Carol Danvers (Brie Larson), aka Captain Marvel. Here's a breakdown of what we see in the trailer:

        * The opening shot shows Carol crashing into a Blockbuster Video store in the '90s, which gives away the time period of the film.
        * We then cut to Carol in her suit, looking fierce and ready for battle.
        * Next, we see Nick Fury (Samuel L. Jackson) without his iconic eye patch, which suggests this might be before he lost his eye.
        * The trailer cuts to a montage of action sequences, showing Carol fighting Kree warriors and destroying enemy ships.",semstamp_lsh,MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ
    2,"Set in the 1990s, Captain Marvel follows Danvers, an Air Force officer, as she becomes one of the universe's most powerful heroes after earth gets involved in a galactic war between alien. The movie is set to release on March 8, 2019.
    Directed by: Anna Boden and Ryan Fleck Written by: Geneva Robertson-Dworet, Nicole Perlman, Meg LeFauve Cast: Brie Larson, Samuel L. Jackson, Ben Mendelsohn, Djimon Hounsou, Lee Pace, Lashana Lynch, Gemma Chan, Annette Bening, Clark Gregg Music by: Pinar Toprak Produced by: Marvel Studios Distributed by: Walt Disney Studios Motion Pictures #CaptainMarvel #BrieLarson #MarvelStudios #AvengersEndgame #MarvelCinematicUniverse #MCU Follow me on Instagram for more updates - instagram.com/the_movie_buff/ Facebook - facebook.com/themoviebuff/
    What if Captain Marvel was introduced earlier in the MCU? My thought process:
    Subscribe Here! 🤩👉 bit.ly/2CQq3oP Join The Discord Server! discord.gg/SyHrD6K Become A Patron! patreon.com/TheFilmTheorist Support Me On Ko-Fi! ko-fi.com/thefilmtheorist #CaptainMarvel #MCU #MarvelStudios #TheFilmTheorist **Copyright Disclaimer Under Section 107 of the Copyright Act 1976, allowance is made for ""fair use"" for purposes such as criticism, comment, news reporting, teaching, scholarship, and research.",semstamp_lsh,MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ
    3,"Set in the 1990s, Captain Marvel follows Danvers, an Air Force officer, as she becomes one of the universe's most powerful heroes after earth gets involved in a galactic war between alien. Here are some fun facts about the movie.

    **1. The Movie Was Written and Directed by Anna Boden and Ryan Fleck**: Boden is the first female director to helm a solo superhero film in the Marvel Cinematic Universe (MCU). She previously directed It's Kind of a Funny Story (2010) and Mississippi Grind (2015).

    **2. Brie Larson Trained for Nine Months to Prepare for the Role**: The actress underwent intense physical training, including martial arts, stunt training, and weightlifting, to play Carol Danvers/Captain Marvel. Her hard work paid off – she can do most of her own stunts!

    **3. The Film Takes Inspiration from the Comics**: The story draws inspiration from Kelly Sue DeConnick's comic book run, particularly the 2014 series Captain Marvel Vol. 7: Earth's Mightiest Hero. Fans of the comics will recognize nods to iconic characters, like Goose the cat.

    **4. Skrulls Were Created Using Advanced VFX Technology**: To bring the shapeshifting aliens to life, visual effects company Digital Domain developed cutting-edge technology to create realistic facial expressions and movements. No prosthetics or makeup were used for the Skrull designs.

    **5. Samuel L. Jackson's Nick Fury Has a Few Surprises**: In this '90s-set film, Nick Fury is a younger, more optimistic agent."
    """

    text_mutator = MaskFillMutator()

    start = time.time()
    mutated_text = text_mutator.mutate(text, num_replacements=0.05)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()