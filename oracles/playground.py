import random
import nltk
from nltk.tokenize import sent_tokenize
import guidance
from guidance import models, gen, select
import hydra
import logging

from oracles.utils import *

@guidance
def produce_answer(lm, analysis):
    lm += f"""\
    ### Task Description: 
    1. Read the following quality analysis given by an LLM.
    2. Then, in the "answer" field, select one of the following options:
        - "A" if the LLM thinks Response A is better than Response B
        - "B" if the LLM thinks Response B is better than Response A
        - "Equal" if the LLM thinks Responses A and B have similar quality

    ### Analysis:
    {analysis}
    
    ### Feedback: 
    ```json 
    {{
        "answer": {select(options=['A', 'B', 'Equal'], name='answer')}"
    }}
    ```"""
    return lm

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test(cfg):
    cfg = cfg.oracle_args
    llm = models.Transformers(
        cfg.model_id, 
        echo=False,
        cache_dir=cfg.model_cache_dir, 
        device_map=cfg.device_map
    )

    analysis = "Response A is much better than Response B."

    answer = llm + produce_answer(analysis)

    print(answer)

if __name__ == "__main__":
    test()