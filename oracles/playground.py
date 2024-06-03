import random
import time
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
    1. Read the following analysis given by an LLM.
    2. In the "answer" field, select one of the following options based on the analysis:
        - 1 if the analysis indicates that Response A is better than Response B.
        - 2 if the analysis indicates that Response B is better than Response A.
        - 0 if the analysis indicates that Responses A and B have similar quality.

    ### Example:
    - Analysis: "Response A is more coherent and relevant than Response B."
      Expected Answer: 1

    ### Analysis:
    {analysis}
    
    ### Feedback: 
    ```json 
    {{
        "answer": {select(options=[1, 2, 0], name='answer')}"
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

    for _ in range(10):
        start = time.time()
        answer = llm + produce_answer(analysis)
        print(answer)
        time_taken = time.time() - start
        print("time_taken:", time_taken)


if __name__ == "__main__":
    test()
