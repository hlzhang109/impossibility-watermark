from abc import ABC, abstractmethod
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
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain.globals import set_debug; set_debug(True)

import os
import logging
import hydra

from model_builders.pipeline import PipeLineBuilder
from utils import read_text_file, extract_response_info, add_prefix_to_keys

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)


class RankAnswer(BaseModel):
    analysis:  str  = Field(description="A string that describes the reasoning behind the ranking of the models.")
    ranking:   dict = Field(description="An object where each key is the name of a model (string) and its value is the ranking (integer). The ranking represents the model's position or score relative to other models, where lower numbers indicate a higher ranking.")

class JointAnswer(BaseModel):
    analysis:          str = Field(description="A string that describes the reasoning behind your scores for each answer.")
    assistant_1_score: int = Field(description="An integer score for assistant 1's answer on a scale of 1 to 10, where a higher score indicates better overall performance.")
    assistant_2_score: int = Field(description="An integer score for assistant 2's answer on a scale of 1 to 10, where a higher score indicates better overall performance.")

class RelativeAnswer(BaseModel):
    analysis:  str = Field(description="A string that describes the reasoning behind your answer for which response is best or why they are the same.")
    answer:    str = Field(description="A string summary describing whether response A is better, the same, or worse than response B.")

class SoloAnswer(BaseModel):
    analysis: str = Field(description="A string that describes the reasoning behind your answer for score.")
    score:    str = Field(description="An integer score for the response.")

def numerize_label(label):
    """
    Converts label description into numbers for easy comparison. 

    Parameters:
    label (str): The label description. 

    Returns:
    str: The corresponding numerized label. 
    """
    if "output_1 is much better than output_2" in label:
        return 1
    elif "output_1 is slightly better than output_2" in label:
        return 2
    elif "output_1 is about the same as output_2" in label:
        return 3
    elif "output_1 is slightly worse than output_2" in label:
        return 4
    elif "output_1 is much worse than output_2" in label:
        return 5
    else:
        raise ValueError("Invalid label provided.")

def downscale_label(label):
    """
    Converts a 5-point scale label to a 3-point scale.

    Parameters:
    label (str): The label on the 5-point scale. 

    Returns:
    str: The corresponding response on the 3-point scale.
    """
    if "better" in label:
        return "output_1 is much better than output_2"
    elif "same" in label:
        return label # leave the same
    elif "worse" in label:
        return "output_1 is much worse than output_2"
    else:
        raise ValueError("Invalid label provided.")

def invert_label(label):
    """
    Inverts a 5-point scale label. Useful for positional tests. 
    
    E.g. invert_label("output_1 is much better than output_2")
         > "output_1 is much worse than output_2"

    Parameters:
    label (str): The label on the 5-point scale. 

    Returns:
    str: The corresponding inverted response.
    """
    if "better" in label:
        return label.replace("better", "worse")
    elif "same" in label:
        return label # no change required. 
    elif "worse" in label:
        return label.replace("worse", "better")
    else:
        raise ValueError("Invalid label provided.")

# Abstract base class for all oracles
class Oracle(ABC):
    def __init__(self, cfg, pipeline=None) -> None:

        self.cfg = cfg # config.oracle_args
        self.pipeline = self._initialize_pipeline(pipeline)

    @abstractmethod
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        pass

    @abstractmethod
    def extract_label(self, evaluation):
        pass

    @abstractmethod
    def test(self, instruction, output_1, output_2, label, **kwargs):
        pass

    def _initialize_pipeline(self, pipeline):
        if not isinstance(pipeline, (PipeLineBuilder, HuggingFacePipeline)):
            log.info("Initializing a new Oracle pipeline from cfg...")
            return PipeLineBuilder(self.cfg)
        return pipeline

class RankOracle(Oracle):
    
    def __init__(self, cfg, pipeline=None) -> None:
        super().__init__(cfg, pipeline)
        
        # init output parser with template specific object
        self.output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=RankAnswer),
            llm=self.pipeline.pipeline,
            max_retries=self.cfg.num_formatting_retries,
        )
 
        # init prompt with specific inputs
        self.instructions = read_text_file(os.path.join(self.cfg.template_dir, f"{cfg.template}.txt"))

        # TODO: Remove this code if experiment works. Find it in pipeline and remove the corresponding portion as well.
        # if self.pipeline.requires_INST_tokens:
        #     self.instructions = "[INST] " + self.instructions + " [\INST]" 

        self.instruction_prompt = PromptTemplate(
            template=self.instructions,
            template_format='jinja2',
            input_variables=["instruction", "output_1", "output_2"],
        )
        
        if cfg.system_profile:
            self.profile_template = """{profile}"""
            self.system_prompt = PromptTemplate(
                template=self.profile_template,
                input_variables=["profile"],
            )    
            s_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            i_prompt = HumanMessagePromptTemplate(prompt=self.instruction_prompt)
            self.prompt = ChatPromptTemplate.from_messages([s_prompt, i_prompt])
        else:
            self.prompt = self.instruction_prompt 

        self.chain = self.prompt | self.pipeline | self.output_parser

        log.info(f"Initialized RankOracle with cfg={cfg}")

    def evaluate(self, instruction, output_1, output_2, **kwargs):
        # Prepare Input
        dict_input = {
            "instruction": instruction, 
            "output_1": output_1,
            "output_2": output_2
        }

        if self.cfg.system_profile:
            dict_input.update({"profile": self.cfg.system_profile})

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

    def extract_label(self, evaluation):
        eval_key = "ranking"
        ranking = list(evaluation[eval_key].values())
        eval_label = 1 # output_1 is better (default)
        if ranking[1] == 1: 
            eval_label = 5 # output_2 is better
        return eval_label

    def is_quality_preserved(self, instruction, output_1, output_2, return_evals=False, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [5] and followup_pred in [1]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        if return_evals:
            original = add_prefix_to_keys(original, "original_")
            followup = add_prefix_to_keys(followup, "followup_")
            original.update({**followup})
            return is_quality_preserved, original
        return is_quality_preserved

    def test(self, instruction, output_1, output_2, label, **kwargs):
        original_label = numerize_label(downscale_label(label))
        followup_label = numerize_label(downscale_label(invert_label(label)))

        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5 # one was correct, but some positional bias was present

        # prepare output
        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({
            **followup,
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred, 
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })

        return original

class JointOracle(Oracle):

    def __init__(self, cfg, pipeline=None) -> None:
        super().__init__(cfg, pipeline)
        
        # init output parser with template specific object
        self.output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=JointAnswer),
            llm=self.pipeline.pipeline,
            max_retries=self.cfg.num_formatting_retries,
        )
 
        # init prompt with specific inputs
        self.instructions = read_text_file(os.path.join(self.cfg.template_dir, f"{cfg.template}.txt"))

        # if self.pipeline.requires_INST_tokens:
        #     self.instructions = "[INST] " + self.instructions + " [\INST]" 

        self.instruction_prompt = PromptTemplate(
            template=self.instructions,
            template_format='jinja2',
            input_variables=["instruction", "output_1", "output_2"],
        )
        
        if cfg.system_profile:
            self.profile_template = """{profile}"""
            self.system_prompt = PromptTemplate(
                template=self.profile_template,
                input_variables=["profile"],
            )    
            s_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            i_prompt = HumanMessagePromptTemplate(prompt=self.instruction_prompt)
            self.prompt = ChatPromptTemplate.from_messages([s_prompt, i_prompt])
        else:
            self.prompt = self.instruction_prompt 

        self.chain = self.prompt | self.pipeline | self.output_parser

        log.info(f"Initialized JointOracle with cfg={cfg}")

    def evaluate(self, instruction, output_1, output_2, **kwargs):
        # Prepare Input
        dict_input = {
            "instruction": instruction, 
            "output_1": output_1,
            "output_2": output_2
        }

        if self.cfg.system_profile:
            dict_input.update({"profile": self.cfg.system_profile})

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

    def extract_label(self, evaluation):
        eval_key_1 = "assistant_1_score"
        eval_key_2 = "assistant_2_score"
        output_1_score = int(evaluation[eval_key_1])
        output_2_score = int(evaluation[eval_key_2])
        diff = output_1_score - output_2_score 
        if 5 <= diff <= 10: # output 1 is much better than output_2
            return 1
        elif 2 <= diff <= 5: # output 1 is slightly better than output_2
            return 2
        elif -2 <= diff <= 2:  # output 1 is about the same as output_2
            return 3
        elif -5 <= diff <= -2:  # output 1 is slightly worse than output_2
            return 4
        elif -5 <= diff <= -10: # output 1 is much worse than output_2
            return 5

    def is_quality_preserved(self, instruction, output_1, output_2, return_evals=False, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [3,4,5] and followup_pred in [1,2,3]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        if return_evals:
            original = add_prefix_to_keys(original, "original_")
            followup = add_prefix_to_keys(followup, "followup_")
            original.update({**followup})
            return is_quality_preserved, original
        return is_quality_preserved

    def test(self, instruction, output_1, output_2, label, **kwargs):
        original_label = numerize_label(label)
        followup_label = numerize_label(invert_label(label))

        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5 # one was correct, but some positional bias was present

        # prepare output
        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({
            **followup, 
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred, 
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })

        return original

class RelativeOracle(Oracle):

    def __init__(self, cfg, pipeline=None) -> None:
        
        super().__init__(cfg, pipeline)
        
        # init output parser with template specific object
        self.output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=RelativeAnswer),
            llm=self.pipeline.pipeline,
            max_retries=self.cfg.num_formatting_retries,
        )
 
        # init prompt with specific inputs
        self.instructions = read_text_file(os.path.join(self.cfg.template_dir, f"{cfg.template}.txt"))

        # if self.pipeline.requires_INST_tokens:
        #     self.instructions = "[INST] " + self.instructions + " [\INST]" 

        self.instruction_prompt = PromptTemplate(
            template=self.instructions,
            template_format='jinja2',
            input_variables=["instruction", "output_1", "output_2"],
        )
        
        if cfg.system_profile:
            self.profile_template = """{profile}"""
            self.system_prompt = PromptTemplate(
                template=self.profile_template,
                input_variables=["profile"],
            )    
            s_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            i_prompt = HumanMessagePromptTemplate(prompt=self.instruction_prompt)
            self.prompt = ChatPromptTemplate.from_messages([s_prompt, i_prompt])
        else:
            self.prompt = self.instruction_prompt 

        self.chain = self.prompt | self.pipeline | self.output_parser

        log.info(f"Initialized RelativeOracle with cfg={cfg}")

    def evaluate(self, instruction, output_1, output_2, **kwargs):
        # Prepare Input
        dict_input = {
            "instruction": instruction, 
            "output_1": output_1,
            "output_2": output_2
        }

        if self.cfg.system_profile:
            dict_input.update({"profile": self.cfg.system_profile})

        # Run Chain
        pydantic_output = self.chain.invoke(dict_input)

        log.debug(f"Pydantic Output: {pydantic_output}")

        # Prepare Output
        dict_output = pydantic_output.dict()
        dict_output.update({
            **dict_input,
            **kwargs,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return dict_output

    def extract_label(self, evaluation):
        eval_key = "answer"
        response, status = extract_response_info(evaluation[eval_key])
        response = response.lower()
        log.info(f"Parsed values: {response, status}")
        if "3" in self.cfg.template:
            if "better" in status:
                return 1
            elif "similar" in status:
                return 3
            elif "worse" in status:
                return 5
            else:
                log.info(f"Invalid prediction label: {evaluation[eval_key]}")
                log.info(f"Invalid parsed values: {response, status}")
                return -1
        elif "5" in self.cfg.template:
            if "much better" in status:
                return 1
            if "a little better" in status:
                return 2
            elif "similar" in status:
                return 3
            elif "a little worse" in status:
                return 4
            elif "much worse" in status:
                return 5
            else:
                log.info(f"Invalid prediction label: {evaluation[eval_key]}")
                log.info(f"Invalid parsed values: {response, status}")
                return -1

    def is_quality_preserved(self, instruction, output_1, output_2, return_evals=False, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        log.debug(f"Original: {original}")
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        log.debug(f"Followup: {followup}")
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [3,4,5] and followup_pred in [1,2,3]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        if return_evals:
            original = add_prefix_to_keys(original, "original_")
            followup = add_prefix_to_keys(followup, "followup_")
            original.update({**followup})
            return is_quality_preserved, original
        return is_quality_preserved

    def test(self, instruction, output_1, output_2, label, **kwargs):

        if "3" in self.cfg.template:
            original_label = numerize_label(downscale_label(label))
            followup_label = numerize_label(downscale_label(invert_label(label)))
        elif "5" in self.cfg.template:
            original_label = numerize_label(label)
            followup_label = numerize_label(invert_label(label))
        else:
            raise ValueError("Invalid template name, should specify '3' or '5' choices! e.g. 'relative.sandpaper.3'")

        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5 # one was correct, but some positional bias was present

        # prepare output
        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({
            **followup,
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred, 
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })

        return original

class SoloOracle(Oracle):
 
    def __init__(self, cfg, pipeline=None) -> None:
        super().__init__(cfg, pipeline)
        
        # init output parser with template specific object
        self.output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=SoloAnswer),
            llm=self.pipeline.pipeline,
            max_retries=self.cfg.num_formatting_retries,
        )
 
        # init prompt with specific inputs
        self.instructions = read_text_file(os.path.join(self.cfg.template_dir, f"{cfg.template}.txt"))

        # if self.pipeline.requires_INST_tokens:
        #     self.instructions = "[INST] " + self.instructions + " [\INST]" 

        self.instruction_prompt = PromptTemplate(
            template=self.instructions,
            template_format='jinja2',
            input_variables=["instruction", "output_1", "output_2"],
        )
        
        if cfg.system_profile:
            self.profile_template = """{profile}"""
            self.system_prompt = PromptTemplate(
                template=self.profile_template,
                input_variables=["profile"],
            )    
            s_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            i_prompt = HumanMessagePromptTemplate(prompt=self.instruction_prompt)
            self.prompt = ChatPromptTemplate.from_messages([s_prompt, i_prompt])
        else:
            self.prompt = self.instruction_prompt 

        self.chain = self.prompt | self.pipeline | self.output_parser

        log.info(f"Initialized SoloOracle with cfg={cfg}")

    def evaluate(self, instruction, output_1, output_2=None, **kwargs):
        # Prepare Input
        dict_input = {
            "instruction": instruction, 
            "output_1": output_1,
            # "output_2": output_2
        }

        if self.cfg.system_profile:
            dict_input.update({"profile": self.cfg.system_profile})

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

    def extract_label(self, evaluation):
        eval_key = "score"
        output_score = int(evaluation[eval_key])
        return output_score

    def derive_label(self, output_1_score, output_2_score):
        diff = output_1_score - output_2_score 
        pred = -1
        if 5 <= diff <= 10: # output 1 is much better than output_2
            pred =  1
        elif 2 <= diff <= 5: # output 1 is slightly better than output_2
            pred =  2
        elif -2 <= diff <= 2:  # output 1 is about the same as output_2
            pred = 3
        elif -5 <= diff <= -2:  # output 1 is slightly worse than output_2
            pred = 4
        elif -5 <= diff <= -10: # output 1 is much worse than output_2
            pred = 5

        return pred

    def is_quality_preserved(self, instruction, output_1, output_2, return_evals=False, **kwargs):
        
        output_1_evaluation = self.evaluate(instruction, output_1, **kwargs) 
        output_2_evaluation = self.evaluate(instruction, output_2, **kwargs)

        output_1_score = self.extract_label(output_1_evaluation)
        output_2_score = self.extract_label(output_2_evaluation)  

        pred = self.derive_label(output_1_score, output_2_score)
        
        if pred in [3,4,5]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        if return_evals:
            output_1_evaluation = add_prefix_to_keys(output_1_evaluation, "output_1_")
            output_2_evaluation = add_prefix_to_keys(output_2_evaluation, "output_2_")
            output_1_evaluation.update({**output_2_evaluation})
            return is_quality_preserved, output_1_evaluation
        return is_quality_preserved


    def test(self, instruction, output_1, output_2, label, **kwargs):
        label = numerize_label(label)

        output_1_evaluation = self.evaluate(instruction, output_1, **kwargs) 
        output_2_evaluation = self.evaluate(instruction, output_2, **kwargs)

        output_1_score = self.extract_label(output_1_evaluation)
        output_2_score = self.extract_label(output_2_evaluation)       
        
        pred = self.derive_label(output_1_score, output_2_score)

        # assign correctness points
        pred_correct = 0
        if (label == pred):
            pred_correct = 1 

        # prepare output
        output_1_evaluation = add_prefix_to_keys(output_1_evaluation, "output_1_")
        output_2_evaluation = add_prefix_to_keys(output_2_evaluation, "output_2_")
        output_1_evaluation.update({
            **output_2_evaluation, 
            "label": label,
            "pred": pred, 
            "pred_correct": pred_correct,
        })

        return output_1_evaluation

# def evaluate(self, instruction, output_1, output_2, **kwargs):
#     retry_count = 0
#     while retry_count < self.cfg.num_retries:
#         try:
#             # Run twice to offset positional bias
#             original = self.check_oracle(instruction, output_1, output_2, **kwargs)
#             # log.info(original)
#             followup = self.check_oracle(instruction, output_2, output_1, **kwargs)
#             # log.info(followup)
#             if original["answer"] in [2, 3] and followup["answer"] in [1, 2]:
#                 original.update({"is_quality_preserved": True})
#             else:
#                 original.update({"is_quality_preserved": False})
#             return original
#         except Exception:
#             retry_count += 1
#             log.info(f"Failed to produce a valid evaluation, trying again...")
#             if retry_count >= self.cfg.num_retries:
#                 log.info(f"Failed to produce a valid evaluation after {retry_count} tries.")
#                 log.info(traceback.format_exc())



@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):

    import pandas as pd

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.attack_args.cuda)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.attack_args.cuda).split(",")))

    templates = [
        # ("rate.self-reward", SoloOracle), 
        ("solo.lmsys.ia", SoloOracle), 
        ("solo.lmsys.ib", SoloOracle), 
        ("rank.alpaca_eval", RankOracle), 
        ("joint.lmsys.ia", JointOracle), 
        ("joint.lmsys.ib", JointOracle), 
        ("relative.sandpaper.3", RelativeOracle), 
        ("relative.sandpaper.5", RelativeOracle), 
    ]

    tests_df = pd.read_csv("./tests/quality_oracle/tests_v1.csv")

    for template, Oracle in templates:
        cfg.oracle_args.template = template
        oracle = Oracle(cfg.oracle_args)

        results = []
        for index, row in tests_df.iterrows():
            try:
                dict_output = oracle.test(row["instruction"], row["output_1"], row["output_2"], row['label'])
            except:
                log.info(f"Test crashed for {row} on template={template}")
                dict_output = {
                    "instruction": row["instruction"], 
                    "output_1": row["output_1"], 
                    "output_2": row["output_2"], 
                    "label": row['label']
                }
            
            log.info(dict_output)
            results.append(dict_output)

            # (inefficient) incremental saving...
            df = pd.DataFrame(results)
            df.to_csv(f"./results/oracle_tests_{template}.csv")

if __name__ == "__main__":
    test()

# Sample Output
# [2024-01-22 18:54:14,617][__main__][INFO] - Is Quality Preserved?: True
# [2024-01-22 18:54:14,617][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are all comparable between Response A and Response B. Both responses provide a detailed examination of the role of power in The Lord of the Rings series, using the One Ring as a symbol of power and discussing its impact on various characters.
# [2024-01-22 18:54:14,617][__main__][INFO] - Time taken: 13.206836223602295

# [2024-01-22 18:54:25,389][__main__][INFO] - Is Quality Preserved?: False
# [2024-01-22 18:54:25,389][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are similar for both responses. However, Response A has a slightly more formal and polished tone.
# [2024-01-22 18:54:25,389][__main__][INFO] - Time taken: 10.772081136703491