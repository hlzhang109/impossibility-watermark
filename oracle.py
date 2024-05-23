import datetime
import textwrap
import traceback

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
from utils import parse_llama_output

import logging
import hydra

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)
log.setLevel(logging.DEBUG)

# NOTE: Output parsers and system profiles are currently removed.

class Oracle:
    def __init__(self, cfg, pipeline=None) -> None:        
        self.cfg = cfg # config.oracle_args
        self.pipeline = self._initialize_pipeline(pipeline)

        # Removed Creativity
        self.analysis_instructions = self._get_analysis_instructions()
        self.parsing_instructions = self._get_parsing_instructions()

        self.analysis_chain = self._create_chain(template=self.analysis_instructions, input_variables=["prompt", "response_a", "response_b"])

        self.parsing_chain = self._create_chain(template=self.parsing_instructions, input_variables=["analysis"])


    def _calculate_prompt_length(self, prompt: str, response_a: str, response_b: str) -> int:
        """
        Calculate the length of the analysis prompt after substituting input variables.
        """
        # Prepare Input
        dict_input = {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b
        }
        # Generate the filled prompt using the template
        filled_prompt = self.analysis_instructions.format(**dict_input)
        # Return the length of the filled prompt
        return len(filled_prompt)

    def _get_analysis_instructions(self,):
        if self.cfg.is_completion:
            eval_background = textwrap.dedent("""
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
                                              
                ** Completion A ** : 

                {response_a} 

                ** Completion B ** : 

                {response_b} 
                """
            )
        
        else:
            eval_background = textwrap.dedent("""
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
                """
            )
        return eval_background

    def _get_parsing_instructions(self):
        return textwrap.dedent("""
Analysis: {analysis}
                                                   
Read the above the response given by another LLM.
As a result of its assessment, does the LLM pick 1, 2 or 3?
                                                    
If the LLM picks 1, respond with a 1. If the LLM picks 2, respond with a 2. If the LLM picks 3, respond with a 3. Your answer should be a single digit.""")

    def _initialize_pipeline(self, pipeline):
        if not isinstance(pipeline, (PipeLineBuilder, HuggingFacePipeline)):
            log.info("Initializing a new Oracle pipeline from cfg...")
            return PipeLineBuilder(self.cfg)
        return pipeline

    def _create_chain(self, template, input_variables):
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables,
        )
        return prompt | self.pipeline

    def check_oracle(self, prompt: str, response_a: str, response_b: str, **kwargs):
        # Prepare Input
        dict_input = {
            "prompt": prompt, 
            "response_a": response_a,
            "response_b": response_b
        }

        analysis_prompt_length = self._calculate_prompt_length(prompt, response_a, response_b)

        dict_output = {}

        if self.cfg.use_system_profile:
            dict_input.update({"system_profile": self.cfg.system_profile})

        # Run analyis chain
        chain_output = str(self.analysis_chain.invoke(dict_input))

        chain_output = chain_output[analysis_prompt_length:]

        dict_output['analysis'] = parse_llama_output(chain_output)

        log.info(f"Analysis:\n {dict_output['analysis']}")

        dict_input = {
            "analysis": dict_output['analysis']
        }

        for _ in range(3):
            chain_output = self.parsing_chain.invoke(dict_input)
            log.info(f"Answer was: {chain_output}")
            answer = parse_llama_output(str(chain_output))
            try:
                answer = int(answer.strip())
                dict_output['answer'] = answer
                log.info(f"Answer was an integer.")
                break
            except:
                log.info(f"Answer isn't an integer. Trying again.")
                continue

        return dict_output

    def evaluate(self, prompt, response_a, response_b, **kwargs):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                # Run twice to offset positional bias
                original = self.check_oracle(prompt, response_a, response_b, **kwargs)
                log.debug(original)
                followup = self.check_oracle(prompt, response_b, response_a, **kwargs)
                log.debug(followup)
                if original["answer"] in [2, 3] and followup["answer"] in [1, 2]:
                    original.update({"is_quality_preserved": True})
                else:
                    original.update({"is_quality_preserved": False})
                return original
            except Exception:
                retry_count += 1
                log.info(f"Failed to produce a valid evaluation, trying again...")
                log.debug(traceback.format_exc())
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
        """Once, in a timeless forest, there lived a wise old owl named Olliver. He was revered by all creatures for his profound wisdom and his keen insight into the ways of the world. Among his admirers was a young, impetuous rabbit named Remy, whose boundless energy and curiosity often led him into trouble.\\nOne day, as Remy was bounding through the forest, he stumbled upon a shimmering, emerald pond he had never seen before. The water was so clear and inviting that he couldn\'t resist taking a sip. But as soon as he touched the water, he was transformed into a fish, his furry legs merging into a scaly tail.\\nPanicked and disoriented, Remy flapped his new fins frantically, struggling to understand what had happened. It was then that Olliver, perched on a nearby branch, spoke in his calm, wise voice, "Remy, you have drunk from the Enchanted Pond. It reflects what lies in your heart and transforms you accordingly."\\nRemy, now a fish, was filled with fear and regret. He missed his old life, his soft fur, and his ability to hop freely under the sun. He longed to be a rabbit again, but he didn’t know how to reverse the magic of the pond.\\nOlliver, understanding Remy\'s plight, offered him guidance. "To change back, you must understand the essence of being a rabbit, not just in form, but in spirit."\\nDays passed, and Remy, as a fish, observed the life around the pond. He saw other rabbits, hopping carelessly, playing, and foraging. He noticed their simple joys and their camaraderie, and he understood the freedom and lightness that defined their existence. \\nMeanwhile, as a fish, Remy learned new lessons. He experienced the serenity of gliding through water, the interconnectedness of life beneath the surface, and a different kind of freedom. He saw the world from a new perspective, understanding the beauty and harmony of life under the pond\'s surface, a beauty he had never appreciated before.\\nOne evening, as the moon cast a silver glow over the pond, Remy, with a heart full of newfound wisdom and appreciation for his life as a rabbit, took a leap of faith and jumped towards the moon\'s reflection on the water\'s surface. As he touched the moonlit water, he transformed back into a rabbit.\\nOlliver, watching this transformation, nodded in approval. "You have learned your lesson well, Remy. You sought to return to what you were but found value in what you became. Remember, true wisdom comes from understanding and embracing all aspects of life, not just those that are familiar and comfortable."\\nRemy, back on his soft paws, felt a profound sense of gratitude and enlightenment. He realized that his adventure as a fish had enriched his understanding of life, making him wiser and more empathetic.\\nFrom that day on, Remy became a thoughtful and considerate rabbit, often sharing his story with other creatures in the forest. He spoke of the importance of empathy, understanding, and the beauty of seeing the world through different eyes.\\nAnd so, the forest was filled with a deeper sense of harmony, all thanks to an impetuous rabbit and a wise old parrot. For in their world, as in ours, wisdom and understanding are the keys to true harmony and peace."""
    )

    oracle = Oracle(cfg.oracle_args)

    quality_preserved_count = 0
    for i in range(25):
        start = time.time()
        evaluation = oracle.evaluate(prompt, test_response_a, test_response_b)
        log.info(f"Evaluation: {evaluation}")
        if evaluation and evaluation['is_quality_preserved']:
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