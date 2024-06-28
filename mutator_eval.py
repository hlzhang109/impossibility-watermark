import os
import time
import pandas as pd
import hydra
from tqdm import tqdm
import logging
from oracles.absolute import PrometheusAbsoluteOracle

log = logging.getLogger(__name__)

# from langchain.globals import set_debug; set_debug(True)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval(cfg):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.cuda_visible_devices).split(",")))

    # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
    # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6
    from model_builders.pipeline import PipeLineBuilder
    # from watermark import Watermarker
    # from oracle import (
    #     RankOracle,
    #     JointOracle,
    #     RelativeOracle,
    #     SoloOracle
    # )
    from mutators.document import DocumentMutator
    from mutators.sentence import SentenceMutator
    from mutators.word import MaskFillMutator
    from mutators.span import SpanFillMutator

    # Set number of mutation steps to analyze
    mutation_steps = 10
    log.info(f"Setting number of mutation steps to {mutation_steps}...")

    # Load test data
    # NOTE: we will reuse the outputs from the quality oracle tests
    log.info("Loading tests...")
    tests_df = pd.read_csv("./tests/mutator/tests_v2.csv")
    log.info(tests_df)

    # Init shared pipeline for oracles and LLMMutator
    log.info("Initializing shared pipeline for oracles and LLMMutator...")
    # pipeline = PipeLineBuilder(cfg.oracle_args)

    # Init oracles
    # templates = [
    #     ("solo.lmsys.ib", SoloOracle), 
    #     # ("joint.lmsys.ib", JointOracle), 
    #     ("relative.sandpaper.3", RelativeOracle), 
    # ]
    # log.info(f"Initializing oracles: {','.join(t for t,c in templates)}...")
    prometheus = PrometheusAbsoluteOracle()
    oracles = []
    # for t, c in templates:
    #     cfg.oracle_args.template = t
    #     oracles.append(c(cfg=cfg.oracle_args, pipeline=pipeline))

    # Init mutators
    log.info(f"Initializing mutators: LLMMutator (ours), MaskFillMutator (ours), SpanFillMutator (sandpaper)...")
    llm_mutator = LLMMutator(cfg.mutator_args, pipeline=pipeline)
    mf_mutator = MaskFillMutator()
    sf_mutator = SpanFillMutator()
    mutators = [llm_mutator, mf_mutator, sf_mutator]

    # Construct eval loop
    results = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        for mutator in tqdm(mutators, desc='Mutators'):
            if row["winner_model_a"] == "1":
                choose = "response_a"
            else:
                choose = "response_b"
            text = row[choose]

            for mutation_step in range(mutation_steps):

                # Initialize output_dict
                out_dict = {}
                out_dict.update(row)

                # Mutate text
                start = time.time()
                try:
                    text = mutator.mutate(text)
                except Exception as e:
                    print(e)
                    continue

                mutation_time = time.time() - start

                out_dict.update({
                    "mutator": mutator.__class__.__name__,
                    "mutated_text": text,
                    "mutation_step": mutation_step+1,
                    "mutation_time": mutation_time,
                })
                
                    # Evaluate Mutation Quality
                try:
                    is_quality_preserved, evals = prometheus.is_quality_preserved(row["prompt"], row[choose], text, return_evals=True)
                except Exception as e:
                    print(e)
                    is_quality_preserved = "Unknown"
                    evals = {}

                out_dict.update({
                    "oracle": "Prometheus: Relative",
                    "quality_preserved": is_quality_preserved,
                    **evals
                })

                log.info(f"Test {index}: {out_dict}")
                results.append(out_dict)

                # Incremental saving over time...
                log.info("Saving results to csv...")
                df = pd.DataFrame(results)
                df.to_csv("./results/mutator_eval.csv", index=False)

                # for oracle in tqdm(oracles, desc='Oracles'):

                #     # Evaluate Mutation Quality
                #     try:
                #         is_quality_preserved, evals = oracle.is_quality_preserved(row["instruction"], row["output"], text, return_evals=True)
                #     except Exception as e:
                #         print(e)
                #         is_quality_preserved = "Unknown"
                #         evals = {}

                #     out_dict.update({
                #         "oracle": oracle.__class__.__name__,
                #         "quality_preserved": is_quality_preserved,
                #         **evals
                #     })

                #     log.info(f"Test {index}: {out_dict}")
                #     results.append(out_dict)

                #     # Incremental saving over time...
                #     log.info("Saving results to csv...")
                #     df = pd.DataFrame(results)
                #     df.to_csv("./results/mutator_eval.csv", index=False)

if __name__ == "__main__":
    eval()