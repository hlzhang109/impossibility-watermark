import hydra
from omegaconf import DictConfig, OmegaConf

import logging

# from generate import TextGenerator
# from watermark import Watermark # try to merge Watermark + TextGenerator from the same file since they share all the same args
# from oracle import Oracle
# from mutate import TextMutator


log = logging.getLogger(__name__)

class AttackTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.watermarker    = Watermark(cfg)
        self.quality_oracle = Oracle(cfg.oracle_args) # expand to incorporate Mixtral
        self.mutator        = TextMutator(cfg.mutator_args, 
                                          self.quality_oracle, 
                                          self.watermarker)

    def attack(self, num_trials, max_steps):
        log.info(f"Starting AttackTester with config: {self.cfg}")
        pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_function(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.mutator_args.model_cache_dir)
    # attacker = AttackTester(cfg)
    # Use AttackTester here...

if __name__ == "__main__":
    my_function()

    # python attack_tester.py watermark_args.watermarking_scheme="umd"
        