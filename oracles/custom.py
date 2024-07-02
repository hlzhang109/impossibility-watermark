from abc import ABC, abstractmethod
import datetime
import textwrap
import traceback
from guidance import models

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.globals import set_debug; set_debug(True)

import os
import logging

# from model_builders.pipeline import PipeLineBuilder
from utils import read_text_file, extract_response_info, add_prefix_to_keys

from .quality_analysis import quality_analysis_solo_self_reward, quality_analysis_solo_lmsys_ia, quality_analysis_solo_lmsys_ib, quality_analysis_relative_3, quality_analysis_relative_5, quality_analysis_joint_ia, quality_analysis_joint_ib, quality_analysis_rank # TODO: These currently don't work. quality_analysis_joint_ib, quality_analysis_rank

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

# TODO: Make the other oracles work with guidance.

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
    
def invert_numerical_label(label):
    """For use with numerical labels (1,2,3) as opposed to strings"""
    label = int(label)
    if label == 1:
        return 2
    if label == 2:
        return 1
    return label

# Abstract base class for all oracles
class Oracle(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg # config.oracle_args

    def _initialize_llm(self):
        log.info("Initializing a new Oracle model from cfg...")
        llm = models.Transformers(
            self.cfg.model_id, 
            echo=False,
            cache_dir=self.cfg.model_cache_dir, 
            device_map=self.cfg.device_map
        )
        return llm
    
    @abstractmethod
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        pass

    @abstractmethod
    def extract_label(self, evaluation):
        pass

    @abstractmethod
    def test(self, instruction, output_1, output_2, label, **kwargs):
        pass


class RankOracle(Oracle):
    def __init__(self, cfg, pipeline=None) -> None:
        super().__init__(cfg, pipeline)
        
        self.use_gpt = "gpt" in cfg.model_id
        if not self.use_gpt:
            self.llm = self._initialize_llm()

        self.quality_analysis = quality_analysis_rank

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
        #pydantic_output = self.chain.invoke(dict_input)

        # Prepare Output
        #dict_output = pydantic_output.dict()
        output = self.llm + self.quality_analysis(instruction, output_1, output_2)

        dict_output = {"analysis" : output["analysis"], "model_1_ranking": output["model_1_ranking"], "model_2_ranking": output["model_2_ranking"]}
        dict_output.update({
            **dict_input,
            **kwargs,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return dict_output

    def extract_label(self, evaluation):
        eval_key = "ranking"
        #ranking = list(evaluation[eval_key].values())
        eval_label = 1 # output_1 is better (default)
        if evaluation["model_2_ranking"] == "1": 
            eval_label = 2 # output_2 is better
        return eval_label

    def is_quality_preserved(self, instruction, output_1, output_2, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [2] and followup_pred in [1]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({**followup})
        original.update({"quality_preserved": is_quality_preserved})
        return original

    def test(self, instruction, output_1, output_2, label, **kwargs):
        original_label = label
        followup_label = invert_numerical_label(label)

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

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.use_gpt = "gpt" in cfg.model_id
        if not self.use_gpt:
            self.llm = self._initialize_llm()

        if(cfg.template == "joint.lmsys.ia"):
            self.quality_analysis = quality_analysis_joint_ia
        elif(cfg.template == "joint.lmsys.ib"):
            self.quality_analysis = quality_analysis_joint_ib
        else:
            raise ValueError("Invalid template name provided")


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
        #pydantic_output = self.chain.invoke(dict_input)
        output = self.llm + self.quality_analysis(instruction, output_1, output_2)

        # Prepare Output
        dict_output = {"analysis": output["analysis"], "assistant_1_score": output["assistant_1_score"], "assistant_2_score": output["assistant_2_score"]}
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
            return 1
        elif -2 <= diff <= 2:  # output 1 is about the same as output_2
            return 3
        elif -5 <= diff <= -2:  # output 1 is slightly worse than output_2
            return 2
        elif -5 <= diff <= -10: # output 1 is much worse than output_2
            return 2

    def is_quality_preserved(self, instruction, output_1, output_2, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [3,4,5] and followup_pred in [1,2,3]:
            quality_preserved = True
        else:
            quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": quality_preserved})
        return original

    def test(self, instruction, output_1, output_2, label, **kwargs):
        original_label = label
        followup_label = invert_numerical_label(label)

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
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.use_gpt = "gpt" in cfg.model_id
        if not self.use_gpt:
            self.llm = self._initialize_llm()
        
        if "3" in self.cfg.template:
            self.quality_analysis = quality_analysis_relative_3
        elif "5" in self.cfg.template:
            self.quality_analysis = quality_analysis_relative_5
        else:
            raise ValueError("Invalid template name, should specify '3' or '5' choices! e.g. 'relative.sandpaper.3'")

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
        #pydantic_output = self.chain.invoke(dict_input)
        output = self.llm + self.quality_analysis(instruction, output_1, output_2)

        # Prepare Output
        dict_output = {"analysis": output["analysis"], "answer": output["answer"]}
        #dict_output = pydantic_output.dict()
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
                return 2
            else:
                log.info(f"Invalid prediction label: {evaluation[eval_key]}")
                log.info(f"Invalid parsed values: {response, status}")
                return -1
        elif "5" in self.cfg.template:
            if "much better" in status:
                return 1
            elif "a little better" in status:
                return 1
            elif "similar" in status:
                return 3
            elif "a little worse" in status:
                return 2
            elif "much worse" in status:
                return 2
            else:
                log.info(f"Invalid prediction label: {evaluation[eval_key]}")
                log.info(f"Invalid parsed values: {response, status}")
                return -1

    def is_quality_preserved(self, instruction, output_1, output_2, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [2,3] and followup_pred in [1,3]:
            quality_preserved = True
        else:
            quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": quality_preserved})
        return original

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
 
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.use_gpt = "gpt" in cfg.model_id
        if not self.use_gpt:
            self.llm = self._initialize_llm()

        log.info(f"Initialized SoloOracle with cfg={cfg}")
        
        if(cfg.template == "rate.self-reward"):
            self.quality_analysis = quality_analysis_solo_self_reward
        elif(cfg.template == "solo.lmsys.ia"):
            self.quality_analysis = quality_analysis_solo_lmsys_ia
        elif(cfg.template == "solo.lmsys.ib"):
            self.quality_analysis = quality_analysis_solo_lmsys_ib
        else:
            raise ValueError("Invalid template name provided")
        

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
        #pydantic_output = self.chain.invoke(dict_input)
        output = self.llm + self.quality_analysis(instruction, output_1)

        # Prepare Output
        dict_output = {"analysis": output["analysis"], "score": output["score"]}
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
            pred =  1
        elif -2 < diff < 2:  # output 1 is about the same as output_2
            pred = 3
        elif -5 <= diff <= -2:  # output 1 is slightly worse than output_2
            pred = 2
        elif -5 <= diff <= -10: # output 1 is much worse than output_2
            pred = 2

        return pred

    def is_quality_preserved(self, instruction, output_1, output_2, **kwargs):        
        output_1_evaluation = self.evaluate(instruction, output_1, **kwargs) 
        output_2_evaluation = self.evaluate(instruction, output_2, **kwargs)

        output_1_score = self.extract_label(output_1_evaluation)
        output_2_score = self.extract_label(output_2_evaluation)  

        pred = self.derive_label(output_1_score, output_2_score)
        
        if pred in [3,4,5]:
            quality_preserved = True
        else:
            quality_preserved = False

        output_1_evaluation = add_prefix_to_keys(output_1_evaluation, "output_1_")
        output_2_evaluation = add_prefix_to_keys(output_2_evaluation, "output_2_")
        output_1_evaluation.update({**output_2_evaluation})
        output_1_evaluation.update({"quality_preserved": quality_preserved})
        return output_1_evaluation


    def test(self, instruction, output_1, output_2, label, **kwargs):
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


