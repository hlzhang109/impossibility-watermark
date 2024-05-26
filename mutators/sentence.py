# from langchain.globals import set_debug
# set_debug(True)

import numpy as np
import traceback
import datetime
import textwrap
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.pydantic_v1 import BaseModel, Field
from utils import diff

import hydra
import logging

log = logging.getLogger(__name__)

class AlteredSentence(BaseModel):
    altered_sentence: str = Field(description="Altered sentence with subtle shifts in meaning")

class SentenceMutator:    
    """
    This class is responsible for loading and initializing an LLM with specified parameters. 
    It also provides methods for mutating a text in a 1- or 2-step process.
    """
    def __init__(self, cfg, pipeline=None):
        self.cfg = cfg
        self.pipeline = pipeline

        # Check if NLTK data is downloaded, if not, download it
        self._ensure_nltk_data()

        # Model Pipeline
        if not isinstance(self.pipeline, HuggingFacePipeline):
            
            # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
            # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6
            from model_builders.pipeline import PipeLineBuilder

            log.info("Initializing a new Text Mutator pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg).pipeline
                
        # Potentially initialize output parser
        self.parser = PydanticOutputParser(pydantic_object=AlteredSentence) if self.cfg.use_pydantic_parser else None
        
        # Prompt Template 
        
       
        self.instructions = textwrap.dedent(
            """
            Rewrite this sentence, introducing subtle shifts in its meaning: {sentence}
            """
        )

        if 'Mixtral' in cfg.model_name_or_path:
            self.instructions = '[INST]\n' + self.instructions + '\n[/INST]'   
        
        self.prompt = PromptTemplate(
            template=self.instructions,
            input_variables=["sentence"],
        )
        
        self.chain = self.prompt | self.pipeline

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')    

    def rephrase_sentence(self, text):
        # Use NLTK to split the text into sentences
        sentences = sent_tokenize(text)
        # Randomly select a sentence
        selected_sentence = random.choice(sentences)
        log.info(f"Sentence to rephrase: {selected_sentence}")

        # Prepare Input
        dict_input = {"sentence": selected_sentence}
        if self.cfg.use_system_profile:
            dict_input.update({"system_profile": self.cfg.system_profile})

        # Generate a creative variation of the sentence
        rephrased_sentence = self.chain.invoke(dict_input)
        log.info(f"Rephrased sentence: {rephrased_sentence}")
        
        creative_sentences = sent_tokenize(rephrased_sentence)
        rephrased_sentence = creative_sentences[0]
        
        log.info(f"Rephrased sentence: {rephrased_sentence}")

        # Replace the original sentence with its creative variation
        sentences[sentences.index(selected_sentence)] = rephrased_sentence
        full_text = ' '.join(sentences)

        return {
            "selected_sentence": selected_sentence,
            "rephrased_sentence": rephrased_sentence, 
            "full_text": full_text,
        }

    def mutate(self, text, return_text_only=True):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                mutated_analysis = self.rephrase_sentence(text)
                if return_text_only:
                    return mutated_analysis["full_text"]
                return mutated_analysis

            except Exception:
                retry_count += 1
                log.info(f"Failed to produce a valid generation, trying again...")
                log.info(traceback.format_exc())
                if retry_count >= self.cfg.num_retries:
                    log.info(f"Failed to produce a valid generation after {retry_count} tries.")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test(cfg):
    import time
    import os

    CUDA_VISIBLE_DEVICES = str(cfg.cuda_visible_devices)
    WORLD_SIZE = str(len(str(cfg.cuda_visible_devices).split(",")))

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

    print(cfg.mutator_args)

    text_mutator = SentenceMutator(cfg.mutator_args)

    start = time.time()
    mutated_text = text_mutator.mutate(text)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Diff: {diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()