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
from langchain_core.pydantic_v1 import BaseModel, Field
from model_builders.pipeline import PipeLineBuilder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from utils import diff

import hydra
import logging

log = logging.getLogger(__name__)

class AlteredSentence(BaseModel):
    altered_sentence: str = Field(description="Altered sentence with subtle shifts in meaning")

class Mutation(BaseModel):
    text:  str = Field(description="Edited text with minimal changes for consistency")

class LLMMutator:    
    """
    This class is responsible for loading and initializing an LLM with specified parameters. 
    It also provides methods for mutating a text in a 1- or 2-step process.
    """
    def __init__(self, cfg, pipeline=None):
        # NOTE: Not working with multiple CUDA devices, so disabled for now.
        # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
                
        self.cfg = cfg # config.mutator_args
        self.pipeline = self._initialize_pipeline(pipeline)
        # If we're using the output parsers, initialize them.
        self.altered_sentence_parser, self.output_parser = self._initialize_pydantic_parsers()
        # Check if NLTK data is downloaded, if not, download it
        self._ensure_nltk_data()

        # Determine the format instructions based on whether we're using the pydantic_parser
        format_instructions_section = self._get_format_instructions_section()

        self.system_profile = """{system_profile}"""

        self.system_prompt = PromptTemplate(
            template=self.system_profile,
            input_variables=["system_profile"],
        )

        # Initialize chains 
        
        # Step 1 - Localize Creativity Injection
        self.step_1_chain = self._create_chain(
            template=self._get_step_1_template(format_instructions_section),
            input_variables=["sentence"],
            partial_variables={"format_instructions": self.altered_sentence_parser._get_format_instructions()} if cfg.use_pydantic_parser else {},
            use_system_profile=cfg.use_system_profile
        )

        # Step 2 - Global Consistency Edit
        self.step_2_chain = self._create_chain(
            template=self._get_step_2_template(format_instructions_section),
            input_variables=["original_text"],
            partial_variables={"format_instructions": self.output_parser._get_format_instructions()} if cfg.use_pydantic_parser else {},
            use_system_profile=cfg.use_system_profile
        )

        # Conditionally add the output parsers if self.cfg.use_pydantic_parser is True
        if self.cfg.use_pydantic_parser:
            self.step_1_chain = self.step_1_chain | self.altered_sentence_parser
            self.step_2_chain = self.step_2_chain | self.output_parser

    def _create_chain(self, template, input_variables, partial_variables, use_system_profile):
        user_prompt = PromptTemplate(template=template, input_variables=input_variables, partial_variables=partial_variables)
        if use_system_profile:
            system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            user_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
            return ChatPromptTemplate.from_messages([system_prompt, user_prompt]) | self.pipeline
        return user_prompt | self.pipeline

    def _get_step_1_template(self, format_instructions_section):
        return textwrap.dedent(f"""
            Rewrite this sentence, introducing subtle shifts in its meaning: {{sentence}}

            {format_instructions_section}
            """)

    def _get_step_2_template(self, format_instructions_section):
        return textwrap.dedent(f"""
            Task: Make minimal edits to this text for consistency and quality. Make sure the text does not get shorter. Only respond with edited text.

            Text:

            {{original_text}} 

            {format_instructions_section}
            """)

    def _get_format_instructions_section(self):
        if self.cfg.use_pydantic_parser:
            return textwrap.dedent("""
                ** Format Instructions **: 
                
                {format_instructions} 
                """)
        return ""

    def _initialize_pipeline(self, pipeline):
        if not isinstance(pipeline, (PipeLineBuilder, HuggingFacePipeline)):
            log.info("Initializing a new Text Mutator pipeline from cfg...")
            return PipeLineBuilder(self.cfg)
        return pipeline

    def _initialize_pydantic_parsers(self):
        if self.cfg.use_pydantic_parser:
            return (PydanticOutputParser(pydantic_object=AlteredSentence),
                    PydanticOutputParser(pydantic_object=Mutation))
        return (None, None)

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')    

    def creatively_alter_sentence(self, text):
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
        rephrased_sentence = self.step_1_chain.invoke(dict_input)
        log.info(f"Rephrased sentence: {rephrased_sentence}")
        # rephrased_sentence = rephrased_sentence.split("[/INST]",1)[1].strip()
        
        creative_sentences = sent_tokenize(rephrased_sentence)
        rephrased_sentence = creative_sentences[0]
        
        log.info(f"Rephrased sentence: {rephrased_sentence}")
        # Replace the original sentence with its creative variation
        sentences[sentences.index(selected_sentence)] = rephrased_sentence
        return ' '.join(sentences)

    def adjust_for_consistency(self, original_text, **kwargs):
        # Prepare Input
        dict_input = {"original_text": original_text}
        if self.cfg.use_system_profile:
            dict_input.update({"system_profile": self.cfg.system_profile})

        # Run Chain
        chain_output = self.step_2_chain.invoke(dict_input)
        
        if self.cfg.use_pydantic_parser:
            dict_output = chain_output.dict()
        else:
            dict_output = {"text" : chain_output}
        
        dict_output.update({
            **dict_input,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        })
        return dict_output

    def mutate(self, text, **kwargs):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                # Step 1: Creatively alter a random sentence
                mutated_text = self.creatively_alter_sentence(text)
                # # Step 1.5: Creatively alter another random sentence
                # mutated_text = self.creatively_alter_sentence(mutated_text)
                # Step 2: Adjust the rest of the text for consistency
                if self.cfg.consistency_check:
                    log.info(f"Running the consistency check...")
                    mutated_text = self.adjust_for_consistency(mutated_text, **kwargs)["text"].strip()
                return mutated_text
            except Exception:
                retry_count += 1
                log.info(f"Failed to produce a valid generation, trying again...")
                if retry_count >= self.cfg.num_retries:
                    log.info(f"Failed to produce a valid generation after {retry_count} tries.")
                    log.info(traceback.format_exc())

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):
    import time
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

    print(cfg.mutator_args)

    text_mutator = LLMMutator(cfg.mutator_args)

    start = time.time()
    mutated_text = text_mutator.mutate(text)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Diff: {diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()