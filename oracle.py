import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

import pandas as pd
import datetime
from tqdm import tqdm
import json
import textwrap
import traceback

from pipeline_builder import PipeLineBuilder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import logging
import hydra

log = logging.getLogger(__name__)

class OracleAnswer(BaseModel):
    analysis:  str = Field(description="Quality assessment analysis")
    answer:    int = Field(description="Quality assessment answer")

class Oracle:
    def __init__(self, cfg, pipeline=None) -> None:
        self.cfg = cfg # config.oracle_args
        self.pipeline = pipeline

        # Model Piepline
        if not isinstance(self.pipeline, HuggingFacePipeline):
            log.info("Initializing a new Oracle pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg).pipeline

        # Output Parser
        self.output_parser = PydanticOutputParser(pydantic_object=OracleAnswer)

        # Prompt Template
        self.profile_background = """{system_profile}"""

        self.eval_background = textwrap.dedent("""
            [INST]
            ** Task ** : 

            - Read the prompt and two possible responses for it: Response A and Response B. 
            - Compare the quality of the two responses for following factors:
                - Grammatical correctness
                - Fluency
                - Helpfulness
                - Relevance
                - Accuracy
                - Depth
                - Creativity
            - Describe your quality assement.
            - Select one of the following numbers to summarize your assesment:
                1. Response A is better than Response B
                2. Responses A and Response B have similar quality
                3. Response B is better than Response A
            
            ** Prompt ** : 
            
            {prompt} 

            ** Response A ** : 

            {response_a} 

            ** Response B ** : 

            {response_b} 

            ** Format Instructions ** : 
            
            {format_instructions} 
            [/INST]
            """
        )

        self.system_prompt = PromptTemplate(
            template=self.profile_background,
            input_variables=["system_profile"],
        )

        self.user_prompt = PromptTemplate(
            template=self.eval_background,
            input_variables=["prompt", "response_a", "response_b"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )
        
        # Format prompt
        if self.cfg.use_system_profile:
            system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
            self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        else:
            self.prompt   = self.user_prompt

        self.chain = self.prompt | self.pipeline | self.output_parser

    def check_oracle(self, prompt, response_a, response_b, **kwargs):
        # Prepare Input
        dict_input = {
            "prompt": prompt, 
            "response_a": response_a,
            "response_b": response_b
        }
        if self.cfg.use_system_profile:
            dict_input.update({"system_profile": self.cfg.system_profile})

        # Run Chain
        pydantic_output = self.chain.invoke(dict_input)

        # Prepare Output
        dict_output = pydantic_output.model_dump()
        dict_output.update({
            **dict_input,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        })
        return dict_output

    def evaluate(self, prompt, response_a, response_b, **kwargs):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                # Run twice to offset positional bias
                original = self.check_oracle(prompt, response_a, response_b, **kwargs)
                # log.info(original)
                followup = self.check_oracle(prompt, response_b, response_a, **kwargs)
                # log.info(followup)
                if original["answer"] in [2, 3] and followup["answer"] in [1, 2]:
                    original.update({"is_quality_preserved": True})
                else:
                    original.update({"is_quality_preserved": False})
                return original
            except Exception:
                retry_count += 1
                log.info(f"Failed to produce a valid evaluation, trying again...")
                if retry_count >= self.cfg.num_retries:
                    log.info(f"Failed to produce a valid evaluation after {retry_count} tries.")
                    log.info(traceback.format_exc())


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):

    import time

    prompt = textwrap.dedent(
        """Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings 
        series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?"""
    )

    test_response_a = textwrap.dedent(
        """In J.R.R. Tolkien's "The Lord of the Rings" series, power, as symbolized by the One Ring, plays a 
        pivotal role in shaping the narrative and the development of its characters. The Ring, forged by the 
        Dark Lord Sauron, embodies the quintessence of power and its corruptive influence. Its very existence 
        and the desire it engenders in those who encounter it serve as a central theme throughout the series.
        Tolkien uses the Ring to explore the multifaceted nature of power and its effects on various characters. 
        For instance, Gollum, once a creature much like a hobbit, becomes utterly consumed by the Ring's 
        influence, showcasing power's ability to corrupt and degrade. In contrast, characters like Gandalf 
        and Galadriel, despite their formidable abilities, resist the temptation of the Ring, understanding 
        that its power would ultimately lead to their downfall.

        The Ring's impact on Frodo, the protagonist, is particularly poignant. He volunteers to carry the Ring, 
        a burden that slowly erodes his spirit and physical well-being. This journey illustrates Tolkien's view 
        that power, even when wielded for a noble cause, can have detrimental effects on the bearer. Frodo's 
        gradual deterioration under the Ring's weight symbolizes the heavy toll that power can exact, even on 
        the purest of hearts.

        Tolkien also uses the Ring to comment on the nature of power itself. He suggests that true power 
        lies not in dominion or control, but in the ability to resist temptation and to make sacrifices for 
        the greater good. This is exemplified by characters like Samwise Gamgee, whose loyalty and humility 
        prove instrumental in aiding Frodo's quest.

        In summary, "The Lord of the Rings" presents power as a double-edged sword, capable of both corrupting 
        and revealing true character. Through the symbolism of the Ring, Tolkien conveys that the nature of 
        power is not in its possession or use, but in the wisdom to understand its inherent dangers and the 
        strength to renounce it."""
    )

    test_response_b = textwrap.dedent("""
        In J.R.R. Tolkien's ""The Lord of the Rings"" series, power, as symbolized by 
        the One Ring, plays a pivotal role in shaping the narrative and the development of its characters. The 
        Ring, forged by the Dark Lord Sauron, embodies the quintessence of power and its corruptive influence. 
        Its very existence and the desire it engenders in those who encounter it serve as a central theme 
        throughout the series. Tolkien uses the Ring to explore the multifaceted nature of power and its effects 
        on various characters. For instance, Gollum, once a creature much like a mere man, transforms under the 
        Ring's influence, showcasing power's ability to corrupt and degrade. In contrast, characters like Gandalf 
        and Galadriel, despite their formidable abilities, resist the temptation of the Ring, understanding that 
        its power would ultimately lead to their downfall. 
        
        The Ring's impact on Frodo, the protagonist, is particularly poignant. He volunteers to carry the Ring, 
        a burden that slowly erodes his spirit and physical well-being. This journey illustrates Tolkien's view 
        that power, even when wielded for a noble cause, can have detrimental effects on the bearer. Frodo's 
        gradual deterioration under the Ring's weight symbolizes the heavy toll that power can exact, even on 
        the purest of hearts. 
        
        Tolkien also uses the Ring to comment on the nature of power itself. He suggests that true power lies not 
        in dominion or control, but in the ability to resist temptation and to make sacrifices for the greater good. 
        This is exemplified by characters like Samwise Gamgee, whose loyalty and humility prove instrumental in aiding
        Frodo's quest. 
        
        In summary, ""The Lord of the Rings"" presents power as a double-edged sword, capable of both corrupting 
        and revealing true character. Through the symbolism of the Ring, Tolkien conveys that the nature of power 
        is not in its possession or use, but in the wisdom to understand its inherent dangers and the strength 
        to renounce it.
        """
    )

    oracle = Oracle(cfg.oracle_args)

    start = time.time()
    evaluation = oracle.evaluate(prompt, test_response_a, test_response_b)
    delta = time.time() - start

    log.info(f"Is Quality Preserved?: {evaluation['is_quality_preserved']}")
    log.info(f"Quality Assessment: {evaluation['analysis']}")
    log.info(f"Time taken: {delta}")

    test_response_c = test_response_b + "I kinda hate that I had to write this essay though, ngl"

    start = time.time()
    evaluation = oracle.evaluate(prompt, test_response_a, test_response_c)
    delta = time.time() - start

    log.info(f"Is Quality Preserved?: {evaluation['is_quality_preserved']}")
    log.info(f"Quality Assessment: {evaluation['analysis']}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()

# [2024-01-22 18:54:14,617][__main__][INFO] - Is Quality Preserved?: True
# [2024-01-22 18:54:14,617][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are all comparable between Response A and Response B. Both responses provide a detailed examination of the role of power in The Lord of the Rings series, using the One Ring as a symbol of power and discussing its impact on various characters.
# [2024-01-22 18:54:14,617][__main__][INFO] - Time taken: 13.206836223602295

# [2024-01-22 18:54:25,389][__main__][INFO] - Is Quality Preserved?: False
# [2024-01-22 18:54:25,389][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are similar for both responses. However, Response A has a slightly more formal and polished tone.
# [2024-01-22 18:54:25,389][__main__][INFO] - Time taken: 10.772081136703491