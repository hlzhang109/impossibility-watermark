import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
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
        
        self.pipeline_builders = {}

        # Helper function to create or reuse pipeline builders.
        def get_or_create_pipeline_builder(model_name_or_path, args):
            if model_name_or_path not in self.pipeline_builders:
                self.pipeline_builders[model_name_or_path] = PipeLineBuilder(args)
            return self.pipeline_builders[model_name_or_path]

        # Create or get existing pipeline builders for generator, oracle, and mutator.
        self.generator_pipe_builder = get_or_create_pipeline_builder(cfg.generator_args.model_name_or_path, cfg.generator_args)
        self.oracle_pipeline_builder = get_or_create_pipeline_builder(cfg.oracle_args.model_name_or_path, cfg.oracle_args)
        self.mutator_pipeline_builder = get_or_create_pipeline_builder(cfg.mutator_args.model_name_or_path, cfg.mutator_args)
        
        # NOTE: We pass the pipe_builder to to watermarker, but we pass the pipeline to the other objects.
        self.watermarker  = Watermarker(cfg, pipeline=self.generator_pipe_builder)
        self.quality_oracle = Oracle(cfg=cfg.oracle_args, pipeline=self.oracle_pipeline_builder.pipeline)
        self.mutator = TextMutator(cfg.mutator_args, pipeline=self.mutator_pipeline_builder.pipeline)
                     

    def count_words(self, text):
        if text is None:
            return 0
        return len(text.split())

    def attack(self, prompt=None, watermarked_text=None):
        """
        Mutate the text for a given number of steps with quality control and watermark checking.
        """
        
        # Prepare the filename for logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d.%H.%M.%S")
        save_path = self.cfg.attack_args.save_name.replace("{time_stamp}", timestamp)

        # Generate watermarked response
        if watermarked_text is None and prompt is not None:
            log.info("Generating watermarked text from prompt...")
            watermarked_text = self.watermarker.generate(prompt)

        assert watermarked_text is not None, "Unable to proceed without watermarked text!"
        
        original_watermarked_text = watermarked_text
        
        watermark_detected, score = self.watermarker.detect(original_watermarked_text)
        
        # Log the original watermarked text
        perturbation_stats = [{
            "step_num": -1,
            "current_text": original_watermarked_text,
            "mutated_text": original_watermarked_text, 
            "current_text_len": self.count_words(original_watermarked_text),
            "mutated_text_len": self.count_words(original_watermarked_text), 
            "quality_preserved": True,
            "watermark_detected": watermark_detected,
            "watermark_score": score,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }]
        save_to_csv(perturbation_stats, save_path)

        # Attack        
        patience = 0
        successful_perturbations = 0
        for step_num in tqdm(range(self.cfg.attack_args.num_steps)):

            if patience > self.cfg.attack_args.patience: # exit after too many failed perturbations
                log.error("Mixing patience exceeded. Attack failed...")
                break

            log.info("Mutating watermarked text...")
            mutated_text = self.mutator.mutate(watermarked_text)
            log.info(f"Mutated text: {mutated_text}")

            current_text_len = self.count_words(watermarked_text)
            mutated_text_len = self.count_words(mutated_text)

            if mutated_text_len / current_text_len < 0.95:
                log.info("Mutation failed to preserve text length requirement...")
                quality_preserved = False
                watermark_detected = True
                score = -1
            else:
                log.info("Checking quality oracle and watermark detector...")
                quality_preserved = self.quality_oracle.evaluate(prompt, original_watermarked_text, mutated_text)['is_quality_preserved']
                if self.cfg.use_watermark:
                    watermark_detected, score = self.watermarker.detect(mutated_text)
                else:
                    watermark_detected, score = False, False
            
            perturbation_stats = [{
                "step_num": step_num, 
                "current_text": watermarked_text,
                "mutated_text": mutated_text, 
                "current_text_len": self.count_words(watermarked_text),
                "mutated_text_len": self.count_words(mutated_text), 
                "quality_preserved": quality_preserved,
                "watermark_detected": watermark_detected,
                "watermark_score": score,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }]
            save_to_csv(perturbation_stats, save_path)
            
            if quality_preserved:
                log.info(f"Mutation successful. This was the {successful_perturbations}th successful perturbation.")
                patience = 0
                successful_perturbations += 1
                watermarked_text = mutated_text
            else:
                # If quality is not maintained, increment patience and retry the mutation process
                log.info("Low quality mutation. Retrying step...")
                patience += 1
                continue

            if watermark_detected:
                log.info("Successul mutation, but watermark still intact. Taking another mutation step..")
                continue
            else:
                log.info("Watermark not detected.")
                if (self.cfg.use_watermark and self.cfg.attack_args.stop_at_removal) or successful_perturbations >= self.cfg.attack_args.num_successful_steps:
                    log.info("Attack successful.")
                    break
        
        return mutated_text

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.attack_args.cuda
    
    attacker = Attack(cfg)
    attacked_text = attacker.attack(cfg.attack_args.prompt, cfg.attack_args.watermarked_text)
                
    log.info(f"Prompt: {cfg.attack_args.prompt}")
    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
        