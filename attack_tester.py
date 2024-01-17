import hydra
import logging

from generate import TextGenerator
from watermark import Watermark # try to merge Watermark + TextGenerator from the same file since they share all the same args
from oracle import Oracle
from mutate import TextMutator

log = logging.getLogger(__name__)

class AttackTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.watermarking_scheme = Watermark(cfg.watermark_args)
        self.text_generation_model = TextGenerator(cfg.generator_args)
        self.quality_oracle = Oracle(cfg.oracle_args)
        self.mutator = TextMutator(cfg.mutator_args)

    def attack(self, num_trials, max_steps):
        log.info(f"Starting AttackTester with config: {self.cfg}")
        pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_function(cfg):
    attacker = AttackTester(cfg)
    # Use AttackTester here...

if __name__ == "__main__":
    my_function()

    # python attack_tester.py watermark_args.watermarking_scheme="umd"
        