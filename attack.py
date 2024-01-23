import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from tqdm import tqdm
from pipeline_builder import PipeLineBuilder
from watermark import Watermarker
from oracle import Oracle
from mutate import TextMutator
from utils import save_to_csv

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class Attack:
    def __init__(self, cfg):
        self.cfg = cfg

        self.generator      = PipeLineBuilder(cfg.generator_args)
        self.watermarker    = Watermarker(cfg, pipeline=self.generator)
        self.quality_oracle = Oracle(cfg=cfg.oracle_args, pipeline=self.generator.pipeline)
        self.mutator        = TextMutator(cfg.mutator_args, pipeline=self.generator.pipeline)

    def attack(self, prompt=None, watermarked_text=None):
        """
        Mutate the text for a given number of steps with quality control and watermark checking.
        """
        
        # Prepare the filename for logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_path = self.cfg.attack_args.save_name.replace("{time_stamp}", timestamp)

        # Generate watermarked response
        if watermarked_text is None and prompt is not None:
            log.info("Generating watermarked text from prompt...")
            watermarked_text = self.watermarker.generate(prompt)

        assert watermarked_text is not None, "Unable to proceed without watermarked text!"

        # Attack        
        patience = 0
        for step_num in tqdm(range(self.cfg.attack_args.num_steps)):

            if patience > self.cfg.attack_args.patience: # exit after too many failed perturbations
                log.error("Mixing patience exceeded. Attack failed...")
                break

            log.info("Mutating watermarked text...")
            mutated_text = self.mutator.mutate(watermarked_text)
            log.info(f"Mutated text: {mutated_text}")

            log.info("Checking quality oracle and watermark detector...")
            quality_preserved = self.quality_oracle.evaluate(prompt, watermarked_text, mutated_text)['is_quality_preserved']
            watermark_detected, score = self.watermarker.detect(mutated_text)
            
            perturbation_stats = [{
                "step_num": step_num, 
                "current_text": watermarked_text,
                "mutated_text": mutated_text, 
                "quality_preserved": quality_preserved,
                "watermark_detected": watermark_detected,
                "watermark_score": score,
            }]
            save_to_csv(perturbation_stats, save_path)

            if quality_preserved and watermark_detected:
                log.info("Successul mutation, but watermark still intact. Taking another mutation step..")
                patience = 0 # reset patience
                watermarked_text = mutated_text
                continue
            
            if not quality_preserved:
                # If quality is not maintained, increment patience and retry the mutation process
                log.info("Low quality mutation. Retrying step...")
                patience += 1
                continue
                
            if quality_preserved and not watermark_detected:
                log.info("Watermark not detected. Attack successful.")
                break
        
        return mutated_text

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    
    attacker = Attack(cfg)
    attacked_text = attacker.attack(cfg.attack_args.prompt)
                
    log.info(f"Prompt: {cfg.attack_args.prompt}")
    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
        