import datetime
import textwrap
import traceback

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field

import os
import logging
import hydra

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class OracleAnswer(BaseModel):
    analysis:  str = Field(description="Quality assessment analysis")
    answer:    str = Field(description="Quality assessment answer")

class Oracle:
    def __init__(self, cfg, pipeline=None) -> None:
        
        self.cfg = cfg # config.oracle_args
        self.pipeline = pipeline

        # Model Pipeline
        if not isinstance(self.pipeline, (PipeLineBuilder, ServerBuilder)):
            log.info("Initializing a new Oracle pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg)

        # Output Parser
        self.output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=OracleAnswer),
            llm=self.pipeline,
            max_retries=self.cfg.num_formatting_retries,
        )

        # Prompt Template
        self.instructions = open(os.path.join(self.cfg.template_dir, f"{self.cfg.template}.txt"))

        self.prompt = PromptTemplate(
            template=self.eval_background,
            template_format='jinja2',
            input_variables=["instruction", "response_a", "response_b"],
            # partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        self.chain = self.prompt | self.pipeline | self.output_parser

    def check_oracle(self, instruction, response_a, response_b, **kwargs):
        # Prepare Input
        dict_input = {
            "instruction": instruction, 
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

    def evaluate(self, instruction, response_a, response_b, **kwargs):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                # Run twice to offset positional bias
                original = self.check_oracle(instruction, response_a, response_b, **kwargs)
                # log.info(original)
                followup = self.check_oracle(instruction, response_b, response_a, **kwargs)
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

    oracle = Oracle(cfg.oracle_args)

    quality_preserved_count = 0
    for i in range(25):
        start = time.time()
        evaluation = oracle.evaluate(instruction, test_response_a, test_response_b)
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