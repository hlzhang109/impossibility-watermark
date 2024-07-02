import hydra
import logging

from .custom import SoloOracle, RankOracle, JointOracle, RelativeOracle
from .absolute import PrometheusAbsoluteOracle
from .relative import PrometheusRelativeOracle

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test(cfg):

    import pandas as pd
    import time

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.attack_args.cuda)
    #os.environ["WORLD_SIZE"] = str(len(str(cfg.attack_args.cuda).split(",")))

    templates = [
        #("rate.self-reward", SoloOracle), 
        #("solo.lmsys.ia", SoloOracle), 
        #("solo.lmsys.ib", SoloOracle), 
        #("rank.alpaca_eval", RankOracle), 
        #("joint.lmsys.ia", JointOracle), 
        #("joint.lmsys.ib", JointOracle), 
        #("relative.sandpaper.3", RelativeOracle), 
        #("relative.sandpaper.5", RelativeOracle), 
        #("prometheus_relative", PrometheusRelativeOracle),
        #("prometheus_absolute", PrometheusAbsoluteOracle)
    ]
    


    tests_df = pd.read_csv("./tests/quality_oracle/lmsys_short.csv")
		

    for template, Oracle in templates:
        cfg.oracle_args.template = template
        oracle = Oracle(cfg.oracle_args)

        results = []
        for index, row in tests_df.iterrows():
            start = time.time()
            test_eval = oracle.test(
                instruction=row["prompt"], 
                output_1=row["response_a"], 
                output_2=row["response_b"],
                label=lmsys_row_to_label(row)
            )
            time_taken = time.time() - start
            print("oracle.test")
            print("pred_correct:", test_eval["pred_correct"])
            print("time_taken:", time_taken)

            dict_output = test_eval
            log.info(dict_output)
            results.append(dict_output)

            # (inefficient) incremental saving...
            df = pd.DataFrame(results)
            df.to_csv(f"./results/oracle_tests_{template}.csv")
            

def lmsys_row_to_label(row):
    if row["winner_model_a"]:
        return 1
    if row["winner_model_b"]:
        return 2
    if row["winner_tie"]:
        return 3
    

if __name__ == "__main__":
    test()

# Sample Output
# [2024-01-22 18:54:14,617][__main__][INFO] - Is Quality Preserved?: True
# [2024-01-22 18:54:14,617][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are all comparable between Response A and Response B. Both responses provide a detailed examination of the role of power in The Lord of the Rings series, using the One Ring as a symbol of power and discussing its impact on various characters.
# [2024-01-22 18:54:14,617][__main__][INFO] - Time taken: 13.206836223602295

# [2024-01-22 18:54:25,389][__main__][INFO] - Is Quality Preserved?: False
# [2024-01-22 18:54:25,389][__main__][INFO] - Quality Assessment: Grammatical correctness, fluency, helpfulness, relevance, accuracy, depth, and creativity are similar for both responses. However, Response A has a slightly more formal and polished tone.
# [2024-01-22 18:54:25,389][__main__][INFO] - Time taken: 10.772081136703491