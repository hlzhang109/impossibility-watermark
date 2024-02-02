import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

# from langchain.globals import set_debug
# set_debug(True)

import traceback
import datetime
import textwrap
import random
import difflib
import time
import nltk
from nltk.tokenize import sent_tokenize

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from pipeline_builder import PipeLineBuilder

import hydra
import logging

log = logging.getLogger(__name__)

class Mutation(BaseModel):
    text:  str = Field(description="Edited text with minimal changes for consistency")

class TextMutator:    
    """
    This class is responsible for loading and initializing an LLM with specified parameters. 
    It also provides methods for mutating a text in a 1- or 2-step process.
    """
    def __init__(self, cfg, pipeline=None):

        self.cfg = cfg # config.mutator_args
        self.pipeline = pipeline
        
        # Model Piepline
        if not isinstance(self.pipeline, HuggingFacePipeline):
            log.info("Initializing a new Text Mutator pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg).pipeline

        # Output Parser
        self.output_parser = PydanticOutputParser(pydantic_object=Mutation)

        # Check if NLTK data is downloaded, if not, download it
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Initialize chains 
        
        # Step 1 - Localize Creativity Injection
        self.step_1_template = textwrap.dedent(
            """
            [INST] 
            Rewrite this sentence creatively: {sentence} 
            [/INST]
            """
        )
        self.step_1_prompt = PromptTemplate(
            template=self.step_1_template,
            input_variables=["sentence"],
        )
        self.step_1_chain = self.step_1_prompt | self.pipeline

        # Step 2 - Global Consistency Edit
        self.step_2_profile = """{system_profile}"""

        self.step_2_template = textwrap.dedent("""
            [INST]

            ** Task **:

            Make minimal edits to this text for consistency and quality. Make sure the text doesn’t get shorter.

            ** Text **: 

            {original_text} 

            ** Format Instructions **: 
        
            {format_instructions} 

            [/INST]
            """
        )

        self.system_prompt = PromptTemplate(
            template=self.step_2_profile,
            input_variables=["system_profile"],
        )

        self.user_prompt = PromptTemplate(
            template=self.step_2_template,
            input_variables=["original_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        if self.cfg.use_system_profile:
            system_prompt      = SystemMessagePromptTemplate(prompt=self.system_prompt)
            user_prompt        = HumanMessagePromptTemplate(prompt=self.user_prompt)
            self.step_2_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        else:
            self.step_2_prompt = self.user_prompt

        self.step_2_chain = self.step_2_prompt | self.pipeline | self.output_parser

    def creatively_alter_sentence(self, text):
        # Use NLTK to split the text into sentences
        sentences = sent_tokenize(text)
        # Randomly select a sentence
        selected_sentence = random.choice(sentences)
        # log.info(f"Sentence to rephrase: {selected_sentence}")

        # Generate a creative variation of the sentence
        rephrased_sentence = self.step_1_chain.invoke({"sentence": selected_sentence})
        # log.info(f"Rephrased sentence: {rephrased_sentence}")
        
        creative_sentences = sent_tokenize(rephrased_sentence)
        rephrased_sentence = creative_sentences[0]
        
        # Replace the original sentence with its creative variation
        sentences[sentences.index(selected_sentence)] = rephrased_sentence
        return ' '.join(sentences)

    def adjust_for_consistency(self, original_text, **kwargs):
        # Prepare Input
        dict_input = {"original_text": original_text}
        if self.cfg.use_system_profile:
            dict_input.update({"system_profile": self.cfg.system_profile})

        # Run Chain
        pydantic_output = self.step_2_chain.invoke(dict_input)

        # Prepare Output
        dict_output = pydantic_output.model_dump()
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
                final_text = self.adjust_for_consistency(mutated_text, **kwargs)["text"]
                return final_text
            except Exception:
                retry_count += 1
                log.info(f"Failed to produce a valid generation, trying again...")
                if retry_count >= self.cfg.num_retries:
                    log.info(f"Failed to produce a valid generation after {retry_count} tries.")
                    log.info(traceback.format_exc())

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):

    import time

    text = textwrap.dedent(
        """
        Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
        The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
        Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
        However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
        In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
        """
    )

    text_mutator = TextMutator(cfg.mutator_args)

    start = time.time()
    mutated_text = text_mutator.mutate(text)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()