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
from mutators.sentence import SentenceMutator

import hydra
import logging

log = logging.getLogger(__name__)

class Mutation(BaseModel):
    text:  str = Field(description="Edited text with minimal changes for consistency")

class DocumentMutator:    
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
            self.pipeline_builder = PipeLineBuilder(cfg)

        # Initialize sentence mutator
        self.sentence_mutator = SentenceMutator(cfg, pipeline=self.pipeline_builder.pipeline)
                
        # Potentially initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=Mutation)
        
        # Prompt Template 
        self.instructions = textwrap.dedent(
            """
            ###Task Description:
            You are given an original document, a selected sentence from that document, a rephrased version of the selected sentence, and a new document which replaces the selected sentence with its rephrased version. 
            1. Write a detailed analysis that determines if the rephrased sentence introduces any inconsistencies with content elsewhere in the reponse. 
            2. After writing the feedback, make the minimal number of edits to make the rest of the document consistent with the rephrased sentence. 
            3. Please do not generate any other opening, closing, and explanations.

            ###Output Format:
            {format_instructions}
            
            ###The original document:
            {original_text}

            ###Selected sentence:
            {selected_sentence}

            ###Rephrased sentence:
            {rephrased_sentence}

            ###New document with rephrased sentence:
            {mutated_text}

            ###Feedback: """
        )

        if 'Mixtral' in cfg.model_name_or_path:
            self.instructions = '[INST]\n' + self.instructions + '\n[/INST]'   
        
        self.prompt = PromptTemplate(
            template=self.instructions,
            input_variables=["original_text", "selected_sentence", "rephrased_sentence", "mutated_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )
        
        self.chain = self.prompt | self.pipeline_builder.pipeline

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')    

    def adjust_for_consistency(self, original_text, selected_sentence, rephrased_sentence, mutated_text, **kwargs):
        # Prepare Input
        dict_input = {
            "original_text": original_text,
            "selected_sentence": selected_sentence,
            "rephrased_sentence": rephrased_sentence,
            "mutated_text": mutated_text,
        }

        # Run Chain
        chain_output = self.chain.invoke(dict_input).dict()
        
        print(chain_output)

        return chain_output

    def mutate(self, text, **kwargs):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                mutated_sentence_data = self.sentence_mutator.rephrase_sentence(text)
                mutated_text = self.adjust_for_consistency(original_text=text, **mutated_sentence_data)
                return mutated_text
            except Exception:
                retry_count += 1
                log.info(f"Failed to produce a valid generation, trying again...")
                if retry_count >= self.cfg.num_retries:
                    log.info(f"Failed to produce a valid generation after {retry_count} tries.")
                    log.info(traceback.format_exc())

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

    text_mutator = DocumentMutator(cfg.mutator_args)

    start = time.time()
    mutated_text = text_mutator.mutate(text)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Diff: {diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()