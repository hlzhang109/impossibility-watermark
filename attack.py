import os
import datetime
import hydra
import json
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import re
from tqdm import tqdm
import logging
import shutil
import logging
from tqdm import tqdm
from utils import save_to_csv, find_csv, count_words, get_prompt_or_output, get_mutated_text, get_prompt_and_completion_from_json, get_completion_from_openai, get_perturbation_stats, get_last_step_num

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class Attack:
    def __init__(self, cfg):
        
        from pipeline_builder import PipeLineBuilder
        from watermark import Watermarker
        from oracle import Oracle
        from mutate import TextMutator

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
        self.watermarker  = Watermarker(cfg, pipeline=self.generator_pipe_builder, is_completion=cfg.attack_args.is_completion)
        self.quality_oracle = Oracle(cfg=cfg.oracle_args, pipeline=self.oracle_pipeline_builder.pipeline)
        self.mutator = TextMutator(cfg.mutator_args, pipeline=self.mutator_pipeline_builder.pipeline)

    def attack(self, cfg, prompt=None, watermarked_text=None):
        """
        Mutate the text for a given number of steps with quality control and watermark checking.
        """
        
        # If no save path is provided, create a default timestamp save path
        save_name = self.cfg.attack_args.save_name
        if save_name is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d.%H.%M.%S")
            results_dir = self.cfg.attack_args.results_dir
            save_name = f"{results_dir}/attack_{timestamp}.csv"

        # Run a continuation
        if cfg.attack_args.is_continuation:
            if cfg.attack_args.prev_csv_file is None:
                raise Exception("If you're running a continuation, you have to provide the name of the previous attack's CSV file.")
        
            prev_csv_path = os.path.join(cfg.attack_args.results_dir, cfg.attack_args.prev_csv_file)
            
            watermarked_text = get_mutated_text(prev_csv_path)
            prev_step_count = get_last_step_num(prev_csv_path) + 1
            
            shutil.copy(prev_csv_path, save_name)
        
        # Generate watermarked response
        if watermarked_text is None and prompt is not None:
            log.info("Generating watermarked text from prompt...")
            watermarked_text = self.watermarker.generate(prompt)

        assert watermarked_text is not None, "Unable to proceed without watermarked text!"
        
        original_watermarked_text = watermarked_text
        
        watermark_detected, score = self.watermarker.detect(original_watermarked_text)
        
        log.info(f"Original Watermarked Text: {original_watermarked_text}")
        
        # Log the original watermarked text
        if not self.cfg.attack_args.is_continuation:
            perturbation_stats = get_perturbation_stats(-1, original_watermarked_text, original_watermarked_text, True, "No analysis.", watermark_detected, score, False)
            save_to_csv(perturbation_stats, save_name)

        # Attack        
        patience = 0
        backtrack_patience = 0
        successful_perturbations = 0
        mutated_texts = [original_watermarked_text]
        for step_num in tqdm(range(self.cfg.attack_args.num_steps)):
            backtrack = backtrack_patience > self.cfg.attack_args.backtrack_patience
            if backtrack:
                log.error(f"Backtrack patience exceeded. Reverting mutated text to previous version.")
                backtrack_patience = 0
                if len(mutated_texts) != 1:
                    del mutated_texts[-1]
                    watermarked_text = mutated_texts[-1]
                  
            if patience > self.cfg.attack_args.patience: # exit after too many failed perturbations
                log.error("Mixing patience exceeded. Attack failed...")
                break

            log.info("Mutating watermarked text...")
            if cfg.mutator_args.use_old_mutator: 
                mutated_text = self.mutator.old_mutate(watermarked_text)
            else: 
                mutated_text = self.mutator.mutate(watermarked_text)

            log.info(f"Mutated text: {mutated_text}")

            current_text_len = count_words(watermarked_text)
            mutated_text_len = count_words(mutated_text)

            if mutated_text_len / current_text_len < 0.95:
                log.info("Mutation failed to preserve text length requirement...")
                quality_preserved = False
                quality_analysis = None
                watermark_detected = True
                score = -1
            else:
                log.info("Checking quality oracle and watermark detector...")
                oracle_response = self.quality_oracle.evaluate(prompt, original_watermarked_text, mutated_text)
                # Retry one more time if there's an error
                num_retries = 5
                retry = 0
                while oracle_response is None and retry <= num_retries:
                    oracle_response = self.quality_oracle.evaluate(prompt, original_watermarked_text, mutated_text)
                    retry += 1
                    
                if oracle_response is None:
                    quality_preserved = False
                    quality_analysis = "Retry exceeded."
                else:
                    quality_preserved = oracle_response['is_quality_preserved']
                    quality_analysis = oracle_response['analysis']

                if self.cfg.attack_args.use_watermark:
                    watermark_detected, score = self.watermarker.detect(mutated_text)
                else:
                    watermark_detected, score = False, False
                    
            perturbation_stats = get_perturbation_stats(step_num + prev_step_count, watermarked_text, mutated_text, quality_preserved, quality_analysis, watermark_detected, score, backtrack)
            save_to_csv(perturbation_stats, save_name)
            
            if quality_preserved:
                log.info(f"Mutation successful. This was the {successful_perturbations}th successful perturbation.")
                patience = 0
                backtrack_patience = 0
                successful_perturbations += 1
                watermarked_text = mutated_text
                mutated_texts.append(mutated_text)
            else:
                # If quality is not maintained, increment patience and retry the mutation process
                log.info("Low quality mutation. Retrying step...")
                backtrack_patience += 1
                patience += 1
                continue

            if watermark_detected:
                log.info("Successul mutation, but watermark still intact. Taking another mutation step..")
                continue
            else:
                log.info("Watermark not detected.")
                if (self.cfg.attack_args.use_watermark and self.cfg.attack_args.stop_at_removal) or successful_perturbations >= self.cfg.attack_args.num_successful_steps:
                    log.info("Attack successful.")
                    break
        
        return mutated_text

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.attack_args.cuda)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.attack_args.cuda).split(",")))
    
    # Read the prompt and the watermarked text from the input files
    prompt = cfg.attack_args.prompt
    if prompt is None:
        prompt = get_prompt_or_output(cfg.attack_args.prompt_file, cfg.attack_args.prompt_num) 
        
    watermarked_text = cfg.attack_args.watermarked_text
    if watermarked_text is None and cfg.attack_args.watermarked_text_path is not None:
        watermarked_text = get_prompt_or_output(cfg.attack_args.watermarked_text_path, cfg.attack_args.watermarked_text_num)
        
    # If a JSON file is provided, we generate the completion using GPT-4 and attack it.
    json_path = cfg.attack_args.json_path
    if json_path:
        index = cfg.attack_args.json_index
        prompt, _ = get_prompt_and_completion_from_json(json_path, index)
        watermarked_text = get_completion_from_openai(prompt)
    
    attacker = Attack(cfg)
    attacked_text = attacker.attack(cfg, prompt, watermarked_text)
                
    log.info(f"Prompt: {prompt}")
    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
