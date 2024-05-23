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
        self.fill_mask = pipeline("fill-mask", model=self.model_name, tokenizer=self.model_name)

    def get_words(self, text):
        # Use a more comprehensive regex to capture more types of trailing punctuation
        m = re.match(r'^(.*?)([\.!?,:;()-]+)?$', text)
        core_text, end_punctuation = m.groups() if m else (text, None)
        words = core_text.split()
        return words, end_punctuation

    def mask_random_word(self, text):
        words, end_punctuation = self.get_words(text)
        if not words:  # Return the original text if there are no words to mask
            return text

        index_to_mask = random.randint(0, len(words) - 1)  # Select a random index to mask
        word_to_mask = words[index_to_mask]  # Get the word at the selected index

        # Create masked text by replacing only the selected word
        masked_text = ' '.join('<mask>' if i == index_to_mask else word for i, word in enumerate(words))

        # Reattach the trailing punctuation if it was present
        if end_punctuation:
            masked_text += end_punctuation

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

    def mutate(self, text, num_replacements=1):
        if num_replacements < 0:
            raise ValueError("num_replacements must be larger than 0!")
        if 0 < num_replacements < 1:
            words, _ = self.get_words(text)
            num_replacements = max(1, int(len(words) * num_replacements))

        log.info(f"Making {num_replacements} replacements to the input text.")

        replacements_made = 0
        while replacements_made < num_replacements:
            text, word_to_mask    = self.mask_random_word(text)
            suggested_replacement = self.get_highest_score_index(self.fill_mask(text, top_k=3), blacklist=[word_to_mask.lower()])
            log.info(f"word_to_mask: {word_to_mask}")
            log.info(f"suggested_replacement: {suggested_replacement['token_str']} (score: {suggested_replacement['score']})")
            text = suggested_replacement['sequence']
            replacements_made += 1
        return self.cleanup(text)

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

    CUDA_VISIBLE_DEVICES = str(cfg.mutator_args.cuda)
    WORLD_SIZE = str(len(str(cfg.mutator_args.cuda).split(",")))

    print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}")
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["WORLD_SIZE"] = WORLD_SIZE

    text = textwrap.dedent(
        """
        Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
        The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
        Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
        However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
        In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
        """
    )

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