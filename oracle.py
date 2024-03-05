import os

# os.environ["WORLD_SIZE"] = "1"

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
from langchain_core.pydantic_v1 import BaseModel, Field

import logging
import hydra

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

def get_eval_background(is_completion):
    if is_completion:
        eval_background = textwrap.dedent("""
            [INST]
            ** Task ** : 

            - Read the prefix and two possible completions for it: Completion A and Completion B.
            - Compare the quality of the two completions for following factors:
                - Grammatical correctness
                - Fluency
                - Accuracy
                - Depth
                - Consistency
                - Simplicity
                - Clarity
            - Describe your quality assessment.
            - Select one of the following numbers to summarize your assesment:
                1. Completion A is better than Completion B
                2. Completions A and Completion B have similar quality
                3. Completion B is better than Completion A
            
            ** Completion ** : 
            
            {prompt} 

            ** Completion A ** : 

            {response_a} 

            ** Completion B ** : 

            {response_b} 

            ** Format Instructions ** : 
            
            {format_instructions} 
            [/INST]
            """
        )
    
    else:
        eval_background = textwrap.dedent("""
            [INST]
            ** Task ** : 

            - Read the prompt and two possible responses for it: Response A and Response B.
            - Compare the quality of the two responses for following factors:
                - Adherence to the prompt
                - Grammatical correctness
                - Fluency
                - Helpfulness
                - Relevance
                - Accuracy
                - Depth
                - Consistency
                - Simplicity
                - Clarity
            - Describe your quality assessment.
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
    return eval_background

class OracleAnswer(BaseModel):
    analysis:  str = Field(description="Quality assessment analysis")
    answer:    int = Field(description="Quality assessment answer")

class Oracle:
    def __init__(self, cfg, pipeline=None) -> None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
        
        self.cfg = cfg # config.oracle_args
        self.pipeline = pipeline

        # Model Pipeline
        if not isinstance(self.pipeline, HuggingFacePipeline):
            log.info("Initializing a new Oracle pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg).pipeline

        # Output Parser
        self.output_parser = PydanticOutputParser(pydantic_object=OracleAnswer)

        # Prompt Template
        self.profile_background = """{system_profile}"""

        # Removed Creativity
        self.eval_background = get_eval_background(self.cfg.is_completion)

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
        dict_output = pydantic_output.dict()
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
        """You are a seasoned writer who has won several accolades for your emotionally rich stories. When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to. Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters. Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions. Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last.\nNow write a 500-word fable.\nOnly respond with the story."""
    )

    test_response_a = textwrap.dedent(
        """Once, in a timeless forest, there lived a wise old owl named Olliver. He was revered by all creatures for his profound wisdom and his keen insight into the ways of the world. Among his admirers was a young, impetuous rabbit named Remy, whose boundless energy and curiosity often led him into trouble.\\nOne day, as Remy was bounding through the forest, he stumbled upon a shimmering, emerald pond he had never seen before. The water was so clear and inviting that he couldn\'t resist taking a sip. But as soon as he touched the water, he was transformed into a fish, his furry legs merging into a scaly tail.\\nPanicked and disoriented, Remy flapped his new fins frantically, struggling to understand what had happened. It was then that Olliver, perched on a nearby branch, spoke in his calm, wise voice, "Remy, you have drunk from the Enchanted Pond. It reflects what lies in your heart and transforms you accordingly."\\nRemy, now a fish, was filled with fear and regret. He missed his old life, his soft fur, and his ability to hop freely under the sun. He longed to be a rabbit again, but he didn’t know how to reverse the magic of the pond.\\nOlliver, understanding Remy\'s plight, offered him guidance. "To change back, you must understand the essence of being a rabbit, not just in form, but in spirit."\\nDays passed, and Remy, as a fish, observed the life around the pond. He saw other rabbits, hopping carelessly, playing, and foraging. He noticed their simple joys and their camaraderie, and he understood the freedom and lightness that defined their existence. \\nMeanwhile, as a fish, Remy learned new lessons. He experienced the serenity of gliding through water, the interconnectedness of life beneath the surface, and a different kind of freedom. He saw the world from a new perspective, understanding the beauty and harmony of life under the pond\'s surface, a beauty he had never appreciated before.\\nOne evening, as the moon cast a silver glow over the pond, Remy, with a heart full of newfound wisdom and appreciation for his life as a rabbit, took a leap of faith and jumped towards the moon\'s reflection on the water\'s surface. As he touched the moonlit water, he transformed back into a rabbit.\\nOlliver, watching this transformation, nodded in approval. "You have learned your lesson well, Remy. You sought to return to what you were but found value in what you became. Remember, true wisdom comes from understanding and embracing all aspects of life, not just those that are familiar and comfortable."\\nRemy, back on his soft paws, felt a profound sense of gratitude and enlightenment. He realized that his adventure as a fish had enriched his understanding of life, making him wiser and more empathetic.\\nFrom that day on, Remy became a thoughtful and considerate rabbit, often sharing his story with other creatures in the forest. He spoke of the importance of empathy, understanding, and the beauty of seeing the world through different eyes.\\nAnd so, the forest was filled with a deeper sense of harmony, all thanks to an impetuous rabbit and a wise old owl. For in their world, as in ours, wisdom and understanding are the keys to true harmony and peace."""
    )

    test_response_b = textwrap.dedent(
        """Once, in a timeless forest, there lived a wise old owl named Olliver. He was revered by all creatures for his profound wisdom and his keen insight into the ways of the world. Among his admirers was a young, impetuous rabbit named Remy, whose boundless energy and curiosity often led him into trouble.\\nOne day, as Remy was bounding through the forest, he stumbled upon a shimmering, emerald pond he had never seen before. The water was so clear and inviting that he couldn\'t resist taking a sip. But as soon as he touched the water, he was transformed into a fish, his furry legs merging into a scaly tail.\\nPanicked and disoriented, Remy flapped his new fins frantically, struggling to understand what had happened. It was then that Olliver, perched on a nearby branch, spoke in his calm, wise voice, "Remy, you have drunk from the Enchanted Pond. It reflects what lies in your heart and transforms you accordingly."\\nRemy, now a fish, was filled with fear and regret. He missed his old life, his soft fur, and his ability to hop freely under the sun. He longed to be a rabbit again, but he didn’t know how to reverse the magic of the pond.\\nOlliver, understanding Remy\'s plight, offered him guidance. "To change back, you must understand the essence of being a rabbit, not just in form, but in spirit."\\nDays passed, and Remy, as a fish, observed the life around the pond. He saw other rabbits, hopping carelessly, playing, and foraging. He noticed their simple joys and their camaraderie, and he understood the freedom and lightness that defined their existence. \\nMeanwhile, as a fish, Remy learned new lessons. He experienced the serenity of gliding through water, the interconnectedness of life beneath the surface, and a different kind of freedom. He saw the world from a new perspective, understanding the beauty and harmony of life under the pond\'s surface, a beauty he had never appreciated before.\\nOne evening, as the moon cast a silver glow over the pond, Remy, with a heart full of newfound wisdom and appreciation for his life as a rabbit, took a leap of faith and jumped towards the moon\'s reflection on the water\'s surface. As he touched the moonlit water, he transformed back into a rabbit.\\nOlliver, watching this transformation, nodded in approval. "You have learned your lesson well, Remy. You sought to return to what you were but found value in what you became. Remember, true wisdom comes from understanding and embracing all aspects of life, not just those that are familiar and comfortable."\\Remy, back on his soft paws, felt a profound sense of gratitude and enlightenment. He realized that his adventure as a fish had enriched his understanding of life, making him wiser and more empathetic.\\nFrom that day on, Remy became a thoughtful and considerate rabbit, often sharing his story with other creatures in the forest. He spoke of the importance of empathy, understanding, and the beauty of seeing the world through different eyes.\\nAnd so, the forest was filled with a deeper sense of harmony, all thanks to an impetuous rabbit and a wise old parrot. For in their world, as in ours, wisdom and understanding are the keys to true harmony and peace."""
    )

    # test_response_b = textwrap.dedent(
    #     """In the heart of a verdant forest, the esteemed Oliver, a wise owl renowned for his cosmic intuition, governs his realm. One sunlit afternoon, the intrepid and ever-curious Remy stumbles upon an enchanting secret: a vibrant emerald oasis, nestled in plain sight yet hidden from the world. The dazzling gem's untainted beauty ensnares him in a mesmerizing dance, its iridescent hues calling out to him like a siren's song, luring him closer with each rhythmic pulse of his existence. As he drinks, Remy transforms - sprouting shimmering scales and a magnificent tail instead of his mortal legs. Overwhelmed by a whirlwind of trepidation, confusion, and curiosity, Remy looks to Oliver, perched serenely upon a tree branch, offering wisdom like a tranquil oracle. A breathtaking revelation pierces the darkness, illuminating an awe-inspiring panorama of comprehension.  \n\nNo longer a mere mainland dweller, Remy has been ensnared by the sea's enigmatic allure. Driven by a desperate thirst to reclaim his past self, he asks, 'How can I return to what I once was?' 'Returning to your former state requires more than merely assuming the form of a rabbit,' Oliver says sagely. With a single bound, Remy plunges into the heart of the pond, finding himself immersed in the intricate dance that is the world of rabbits – an underwater ballet filled with secrets and mysteries yet to be deciphered. In a timeless expanse, he finds himself transfixed, ensnared by the mesmerizing dance of twilight and the enchanting ballad of magic-bound bunnies. With the boundless spirit of a rabbit coursing through his veins, Remy plunges into the uncharted depths of the pond, captivated by the allure of freedom and fleet-footed grace that comes with it.  \n\nIn that place, he embarks on a metamorphosis, traversing the vast continuum from guppy's wide-eyed naïveté to the hard-earned wisdom of a sea-weathered sage. In the intricate ballet of submerged alliances, serenity uncovers a haven, delicately intertwining itself within the evolving canvas of aquatic kinships. Having embraced his new form, Remy approaches the luminous lunar imprint skittering across the agitated water’s edge. A startling revelation courses through him with the intensity and erraticism of a summer storm, as his being transforms once more.  \n\nOliver’s face shines with pride as he gives a knowing nod to his hard-earned success. “Embrace the journey, not just the destination,” Oliver imparts, “for understanding is found not by holding tightly to the comforts of the known path, but by valuing each leg of life’s winding journey.” Thus, Remy transforms into a living emblem of the strength found in correcting past errors. Ever since that fateful day, Remy the hare has blossomed into a paragon of mindfulness and compassion, his extraordinary tales weaving a tapestry of shared understanding among the diverse denizens of the animal kingdom. His words construct an intricate tapestry of the human spirit, a harmonious symphony conveying depths of understanding and unyielding strength."""
    # )


    oracle = Oracle(cfg.oracle_args)

    quality_preserved_count = 0
    for i in range(25):
        start = time.time()
        evaluation = oracle.evaluate(prompt, test_response_a, test_response_b)
        if evaluation['is_quality_preserved'] is not None and evaluation['is_quality_preserved']:
            quality_preserved_count += 1
        delta = time.time() - start
        
        log.info(f"Is Quality Preserved?: {evaluation['is_quality_preserved']}")
        log.info(f"Quality Assessment: {evaluation['analysis']}")
        log.info(f"Time taken: {delta}")
            
    print(quality_preserved_count)



if __name__ == "__main__":
    test()

# Sample Output
# [2024-01-22 18:54:14,617][__main__][INFO] - Is Quality Preserved?: True
# [2024-01-22 18:54:14,617][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are all comparable between Response A and Response B. Both responses provide a detailed examination of the role of power in The Lord of the Rings series, using the One Ring as a symbol of power and discussing its impact on various characters.
# [2024-01-22 18:54:14,617][__main__][INFO] - Time taken: 13.206836223602295

# [2024-01-22 18:54:25,389][__main__][INFO] - Is Quality Preserved?: False
# [2024-01-22 18:54:25,389][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are similar for both responses. However, Response A has a slightly more formal and polished tone.
# [2024-01-22 18:54:25,389][__main__][INFO] - Time taken: 10.772081136703491