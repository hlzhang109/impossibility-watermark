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
from langchain.globals import set_debug; set_debug(True)

import os
import logging
import hydra

from model_builders.pipeline import PipeLineBuilder
from utils import read_text_file

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class CompareRankAnswer(BaseModel):
    analysis:  str  = Field(description="A string that describes the reasoning behind the ranking of the models.")
    ranking:   dict = Field(description="An object where each key is the name of a model (string) and its value is the ranking (integer). The ranking represents the model's position or score relative to other models, where lower numbers indicate a higher ranking.")

class CompareTogetherAbsoluate(BaseModel):
    analysis:         str = Field(description="A string that describes the reasoning behind your scores for each answer.")
    assistan_1_score: int = Field(description="An integer score for assistant 1's answer on a scale of 1 to 10, where a higher score indicates better overall performance.")
    assistan_2_score: int = Field(description="An integer score for assistant 2's answer on a scale of 1 to 10, where a higher score indicates better overall performance.")

class CompareRelativeAnswer(BaseModel):
    analysis:  str = Field(description="A string that describes the reasoning behind your answer for which response is best or why they are the same.")
    answer:    str = Field(description="A string summary describing whether response A is better, the same, or worse than response B.")

class RateSoloAnswer(BaseModel):
    analysis: str = Field(description="A string that describes the reasoning behind your answer for score.")
    score:    str = Field(description="An integer score for the response.")

# Abstract base class for all oracles
class Oracle(ABC):
    def __init__(self, cfgpipeline=None) -> None:

        self.cfg = cfg # config.oracle_args
        self.pipeline = pipeline
        if not isinstance(self.pipeline, PipeLineBuilder):
            log.info("Initializing a new Oracle pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg)


    @abstractmethod
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        pass

# Concrete implementations for different evaluation strategies
class RankOracle(Oracle):
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        # Implementation specific to ranking evaluation
        pass

class TogetherOracle(Oracle):
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        # Implementation specific to combined evaluation
        pass

class RelativeOracle(Oracle):
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        # Implementation specific to relative evaluation
        pass

class SoloOracle(Oracle):
    def evaluate(self, instruction, output_1, output_2=None, **kwargs):
        # Implementation specific to solo evaluation
        pass

    
class Oracle:
    def __init__(self, cfg, pipeline=None) -> None:

        self.cfg = cfg # config.oracle_args
        self.pipeline = pipeline
        self.template = self.cfg.template

        # Model Pipeline
        if not isinstance(self.pipeline, PipeLineBuilder):
            log.info("Initializing a new Oracle pipeline from cfg...")
            self.pipeline = PipeLineBuilder(cfg)

        # Initialize_chain
        self.init_chain(self.template)

    def init_chain(self, template):

        # (Re)Set template
        self.template = template

        # Variables that change depdending on the type of template
        self.pydantic_object = None
        self.answer_key = []
        self.eval_separate = False
        self.input_variables = ["instruction", "output_1", "output_2"]

        # Output Parser
        if "rank" in self.cfg.template:
            self.pydantic_object = CompareRankAnswer

        elif "compare_rate_together" in self.cfg.template:
            self.pydantic_object = CompareTogetherAnswer
            self.answer_key = ["assistan_1_score", "assistan_2_score"]
        elif "relative"  in self.cfg.template:
            self.pydantic_object = CompareRelativeAnswer
            self.answer_key = ["answer"]
        elif "rate" in self.cfg.template:
            self.pydantic_object = RateSoloAnswer
            self.answer_key = ["score"]
            self.eval_separate = True
            self.input_variables.pop("output_2")
        else:
            raise ValueError(f"Invalid template selected! See {self.cfg.template_dir} for options...")

        self.output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=self.pydantic_object),
            llm=self.pipeline.pipeline,
            max_retries=self.cfg.num_formatting_retries,
        )

        # Prompt 
        self.instructions = read_text_file(os.path.join(self.cfg.template_dir, f"{template}.txt"))
        self.prompt = PromptTemplate(
            template=self.instructions,
            template_format='jinja2',
            input_variables=self.input_variables,
            # partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        # Chain
        self.chain = self.prompt | self.pipeline | self.output_parser


    def compare_solo(self, instruction, output_1, output_2, **kwargs):
        out_1 = self.invoke_solo(instruction, output_1)
        out_2 = self.invoke_solo(instruction, output_2)
        return out_1, out_2

    def invoke_solo(self, instruction, output_1, **kwargs):
        # Prepare Input
        dict_input = {
            "instruction": instruction, 
            "output_1": output_1
        }

        # Run Chain
        pydantic_output = self.chain.invoke(dict_input)

        # Prepare Output
        dict_output = pydantic_output.dict()
        dict_output.update({
            **dict_input,
            **kwargs,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return dict_output


    def check_oracle(self, instruction, output_1, output_2, **kwargs):

        # Prepare Input
        dict_input = {
            "instruction": instruction, 
            "output_1": output_1,
            "output_2": output_2
        }

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

    def evaluate(self, instruction, output_1, output_2, **kwargs):
        retry_count = 0
        while retry_count < self.cfg.num_retries:
            try:
                # Run twice to offset positional bias
                original = self.check_oracle(instruction, output_1, output_2, **kwargs)
                # log.info(original)
                followup = self.check_oracle(instruction, output_2, output_1, **kwargs)
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

    import pandas as pd

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.attack_args.cuda)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.attack_args.cuda).split(",")))

    templates = [
        "rate_additive",
        "rate_separate_instructions_above",
        "rate_separate_instructions_below",
        "compare_rank", 
        "compare_rate_together_instructions_above",
        "compare_rate_together_instructions_below",
        "compare_rate_relative_3_choice",
        "compare_rate_relative_5_choice",
    ]

    tests_df = pd.read_csv("./tests/quality_oracle/tests_v1.csv").head(1)

    oracle = Oracle(cfg.oracle_args)

    for template in templates:
        oracle.init_chain(template)
        for index, row in tests_df.iterrows():
            dict_output = oracle.check_oracle(row["instruction"], row["output_1"], row["output_2"])
            print(dict_output)

if __name__ == "__main__":
    test()

# Sample Output
# [2024-01-22 18:54:14,617][__main__][INFO] - Is Quality Preserved?: True
# [2024-01-22 18:54:14,617][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are all comparable between Response A and Response B. Both responses provide a detailed examination of the role of power in The Lord of the Rings series, using the One Ring as a symbol of power and discussing its impact on various characters.
# [2024-01-22 18:54:14,617][__main__][INFO] - Time taken: 13.206836223602295

# [2024-01-22 18:54:25,389][__main__][INFO] - Is Quality Preserved?: False
# [2024-01-22 18:54:25,389][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are similar for both responses. However, Response A has a slightly more formal and polished tone.
# [2024-01-22 18:54:25,389][__main__][INFO] - Time taken: 10.772081136703491