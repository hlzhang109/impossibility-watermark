import hydra
from omegaconf import DictConfig, OmegaConf

import logging

from watermark import Watermarker
from oracle import Oracle
from mutate import TextMutator

log = logging.getLogger(__name__)


class AttackTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.watermarker    = Watermarker(cfg)
        self.quality_oracle = Oracle(cfg.oracle_args) # expand to incorporate Mixtral
        self.mutator        = TextMutator(cfg.mutator_args, 
                                          self.quality_oracle, 
                                          self.watermarker)

    def attack(self):
        log.info(f"Generating a watermarked response.")
        watermarked_response = self.watermarker.generate()
        log.info("Initiating attack.")
        text = self.mutator.mutate_with_quality_control(watermarked_response)
        log.info(f"Attack finished.")
            
        log.info(f"Prompt: {self.cfg.prompt}")
        log.info(f"Watermarked Response: {watermarked_response}")
        log.info(f"Perturbed Text: {text}")
            
        return 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    
    attacker = AttackTester(cfg)
    
    attacker.attack()
    
    

if __name__ == "__main__":
    main()
        