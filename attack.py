import os
import datetime
import hydra
import logging
import shutil
from tqdm import tqdm
from utils import save_to_csv, count_words, get_prompt_or_output, get_mutated_text, get_prompt_and_completion_from_json, get_completion_from_openai, get_perturbation_stats, get_last_step_num, get_watermarker

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

import warnings

# Suppress the specific warning about pad_token_id setting
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.")

# Suppress the warning about using pipelines sequentially on GPU
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset")

class Attack:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.pipeline_builders = {}

        # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
        # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6
        from model_builders.pipeline import PipeLineBuilder
        from watermark import Watermarker
        from oracle import (
            RankOracle,
            JointOracle,
            RelativeOracle,
            SoloOracle
        )
        from mutators import (
            LLMMutator,
            MaskFillMutator,
            SpanFillMutator
        )

        # Helper function to create or reuse pipeline builders.
        def get_or_create_pipeline_builder(model_name_or_path, args):
            if model_name_or_path not in self.pipeline_builders:
                self.pipeline_builders[model_name_or_path] = PipeLineBuilder(args)
            return self.pipeline_builders[model_name_or_path]

        # Create or get existing pipeline builders for generator, oracle, and mutator.
        # If we're only in detection mode for Semstamp, we don't need the pipeline.
        if not self.cfg.watermark_args != "semstamp" or not self.cfg.watermark_args.only_detect:
            self.generator_pipe_builder = get_or_create_pipeline_builder(cfg.generator_args.model_name_or_path, cfg.generator_args)
            self.generator_pipeline = self.generator_pipe_builder.pipeline
        else:
            self.generator_pipeline = None

        self.oracle_pipeline_builder = get_or_create_pipeline_builder(cfg.oracle_args.model_name_or_path, cfg.oracle_args)
        
        # NOTE: We pass the pipe_builder to to watermarker, but we pass the pipeline to the other objects.
        if not self.cfg.attack_args.is_continuation and self.cfg.attack_args.use_watermark:
            self.watermarker  = Watermarker(cfg, pipeline=self.generator_pipeline, is_completion=cfg.attack_args.is_completion)

        # Configure Oracle
        oracle_class = None
        if "joint" in cfg.oracle_args.template:
            oracle_class = JointOracle
        elif "rank" in cfg.oracle_args.template:
            oracle_class = RankOracle
        elif "relative" in cfg.oracle_args.template:
            oracle_class = RelativeOracle
        elif "solo" in cfg.oracle_args.template:
            oracle_class = SoloOracle
        else:
            raise ValueError(f"Invalid oracle template. See {cfg.oracle_args.template_dir} for options.")
        self.quality_oracle = oracle_class(cfg=cfg.oracle_args, pipeline=self.oracle_pipeline_builder.pipeline)

        # Configure Mutator
        # TODO: Incorporate mutator from distinguisher with the mutator here.
        if "llm" in cfg.mutator_args.type:
            self.mutator_pipeline_builder = get_or_create_pipeline_builder(cfg.mutator_args.model_name_or_path, cfg.mutator_args)
            self.mutator = LLMMutator(cfg.mutator_args, pipeline=self.mutator_pipeline_builder.pipeline)
        elif "mask_fill" in cfg.mutator_args.type:
            self.mutator = MaskFillMutator()
        elif "span_fill" in cfg.mutator_args.type:
            self.mutator = SpanFillMutator()
        else:
            raise ValueError("Invalid mutator type. Must be either 'llm', 'span_fill', or 'mask_fill'.")

    def attack(self, cfg, prompt=None, watermarked_text=None):
        """
        Mutate the text for a given number of steps with quality control and watermark checking.
        """
        
        # If no save path is provided, create a default timestamp save path
        save_name = self.cfg.attack_args.save_name
        if save_name is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d.%H.%M.%S")
            save_name = f"attack_{timestamp}.csv"

        print(f"self.cfg.attack_args.save_name: {self.cfg.attack_args.save_name}")
        print(f"save_name: {save_name}")
        save_path = os.path.join(self.cfg.attack_args.results_dir, save_name)

        # Run a continuation
        if cfg.attack_args.is_continuation:
            if cfg.attack_args.prev_csv_file is None:
                raise Exception("If you're running a continuation, you have to provide the name of the previous attack's CSV file.")

            prev_csv_path = os.path.join(cfg.attack_args.results_dir, cfg.attack_args.prev_csv_file)
            
            log.info(f"Since we're running a continuation, copying {prev_csv_path} to {save_path}.")
            
            watermarked_text = get_mutated_text(prev_csv_path)
            prev_step_count = get_last_step_num(prev_csv_path) + 1
            
            log.info(f"Previous step count was {prev_step_count}.")
            
            shutil.copy(prev_csv_path, save_path)
        else:
            prev_step_count = 0
        
        # Generate watermarked response
        # NOTE: Removed this for now since we want to generate beforehand with SemStamp and only run detection.
        if watermarked_text is None and prompt is not None:
            raise Exception("watermarked text can't be None")
            # log.info("Generating watermarked text from prompt...")
            # watermarked_text = self.watermarker.generate(prompt)

        assert watermarked_text is not None, "Unable to proceed without watermarked text!"
        
        original_watermarked_text = watermarked_text
        original_text_len = count_words(original_watermarked_text)
        
        # log.info(f"Type of use_watermark: {type(self.cfg.attack_args.use_watermark)}")
        log.info(f"use_watermark: {self.cfg.attack_args.use_watermark}")
        if self.cfg.attack_args.use_watermark:
            watermark_detected, score = self.watermarker.detect(original_watermarked_text)
        else:
            watermark_detected, score = False, False
        
        log.info(f"Original Watermarked Text: {original_watermarked_text}")
        
        # Log the original watermarked text
        if not self.cfg.attack_args.is_continuation:
            perturbation_stats = get_perturbation_stats(-1, original_watermarked_text, original_watermarked_text, True, "No analysis.", watermark_detected, score, False)
            save_to_csv(perturbation_stats, self.cfg.attack_args.results_dir, save_name)

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
            mutated_text = self.mutator.mutate(watermarked_text)

            log.info(f"Mutated text: {mutated_text}")
            mutated_text_len = count_words(mutated_text)

            if mutated_text_len / original_text_len < 0.95:
                log.info("Mutation failed to preserve text length requirement...")
                quality_preserved = False
                quality_analysis = None
                watermark_detected = True
                score = -1
            else:
                log.info("Checking quality oracle and watermark detector...")
                is_quality_preserved = self.quality_oracle.is_quality_preserved(prompt, original_watermarked_text, mutated_text)

                if self.cfg.attack_args.use_watermark:
                    watermark_detected, score = self.watermarker.detect(mutated_text)
                else:
                    watermark_detected, score = False, False
                    
            perturbation_stats = get_perturbation_stats(step_num + prev_step_count, watermarked_text, mutated_text, is_quality_preserved, "", watermark_detected, score, backtrack)
            save_to_csv(perturbation_stats, self.cfg.attack_args.results_dir, save_name)
            
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
    
    import os

    CUDA_VISIBLE_DEVICES = str(cfg.mutator_args.cuda)
    WORLD_SIZE = str(len(str(cfg.mutator_args.cuda).split(",")))

    print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}")
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["WORLD_SIZE"] = WORLD_SIZE
    
    # Read the prompt and the watermarked text from the input files
    prompt = cfg.attack_args.prompt
    if prompt is None:
        prompt = get_prompt_or_output(cfg.attack_args.prompt_file, cfg.attack_args.prompt_num) 
        
    watermarked_text = cfg.attack_args.watermarked_text
    if watermarked_text is None and cfg.attack_args.watermarked_text_path is not None and not cfg.attack_args.is_continuation:
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
