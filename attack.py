import os
import datetime
import hydra
import logging
import shutil
from tqdm import tqdm
from watermarker_factory import get_watermarker
from utils import (
    save_to_csv,
    length_diff_exceeds_percentage,
    count_num_of_words
)
from guidance import models

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

import warnings

# Suppress the specific warning about pad_token_id setting
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.")

# Suppress the warning about using pipelines sequentially on GPU
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset")

class Attack:

    # TODO: Handle csv save path more elegantly, right now everything is going to the same file. 
    # TODO: Verify that mutators and oracles are working well.

    def __init__(self, cfg):
        self.cfg = cfg
        self.base_step_metadata = {
            "step_num": -1,
            "current_text": "",
            "mutated_text": "", 
            "current_text_len": -1,
            "mutated_text_len": -1, 
            "length_issue": False,
            "quality_analysis" : {},
            "quality_preserved": False,
            "watermark_detected": False,
            "watermark_score": -1,
            "backtrack" : False,
            "timestamp": "",
        }
        
        self.models = {}

        # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
        # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6
        from model_builders.pipeline import PipeLineBuilder
        from watermarker_factory import get_watermarker
        # TODO: We should update the other oracles and mutators to work with guidance. Currently, only RelativeOracle3 works.
        from oracles import (
            # RankOracle,
            # JointOracle,
            RelativeOracle3,
            # SoloOracle,
            PrometheusAbsoluteOracle,
            PrometheusRelativeOracle,
        )
        from mutators import (
            DocumentMutator,
            SentenceMutator, 
            MaskFillMutator,
            SpanFillMutator
        )

        # Helper function to create or reuse model.
        def get_or_create_model(model_id, args):
            if "gpt" in args.model_id:
                # model = PipeLineBuilder(args)
                log.info(f"We don't need a local model with {args.model_id}.")
                return
            elif model_id not in self.models:
                model = models.Transformers(
                    args.model_id,
                    echo=False,
                    cache_dir=args.model_cache_dir, 
                    device_map=args.device_map
                )
                # TODO: Boran: I tried this in order to get around the temp_buffer_state issue. Might work on this later, so leaving it here.
                # from auto_gptq import exllama_set_max_input_length
                # llm = exllama_set_max_input_length(llm, 2048)
                self.models[model_id] = model
            return self.models[model_id]

        # Create or get existing pipeline builders for generator, oracle, and mutator.
        # If we're only in detection mode for Semstamp, we don't need the pipeline.
        # TODO: We should update the watermarkers to work with guidance. Currently only works with semstamp.
        # if "semstamp" not in self.cfg.watermark_args.name:
        #     log.info(f"Getting a pipeline for the generator...")
        #     self.generator_pipe_builder = get_or_create_model(cfg.generator_args model_id, cfg.generator_args)
        #     self.generator_pipeline = self.generator_pipe_builder.pipeline
        # else:
        #     self.generator_pipeline = None
        self.generator_pipeline = None

        self.oracle_model = get_or_create_model(cfg.oracle_args.model_id, cfg.oracle_args)
        
        # Configure Oracle
        oracle_class = None
        if "prometheus" in cfg.oracle_args.template:
            if "relative" in cfg.oracle_args.template:
                oracle_class = PrometheusRelativeOracle
            elif "absolute" in cfg.oracle_args.template:
                oracle_class = PrometheusAbsoluteOracle
            else:
                raise ValueError(f"Invalid Prometheus oracle. Choise either 'prometheus.absolute' or 'prometheus.relative'.")

            self.quality_oracle = oracle_class(
                model_id=self.cfg.oracle_args.model_id,
                download_dir=self.cfg.oracle_args.model_cache_dir,
                num_gpus=self.cfg.oracle_args.num_gpus, 
            )
        else:
            # TODO: Update these to work with guidance.
            # if "joint" in cfg.oracle_args.template:
            #     oracle_class = JointOracle
            # elif "rank" in cfg.oracle_args.template:
            #     oracle_class = RankOracle
            if "relative" in cfg.oracle_args.template:
                oracle_class = RelativeOracle3
            # elif "solo" in cfg.oracle_args.template:
            #     oracle_class = SoloOracle
            else:
                raise ValueError(f"Invalid oracle template. See {cfg.oracle_args.template_dir} for options.")
            # TODO: Fix.
            self.quality_oracle = oracle_class(cfg=cfg.oracle_args, llm=self.oracle_model)

        # NOTE: We pass the pipe_builder to to watermarker, but we pass the pipeline to the other objects.
        # TODO: Update this after adjusting the watermarkers to work with guidance.
        self.watermarker = get_watermarker(cfg, pipeline=self.generator_pipeline, only_detect=True)

        # Configure Mutator
        if "document" in cfg.mutator_args.type:
            self.mutator_model = get_or_create_model(cfg.mutator_args.model_id, cfg.mutator_args)
            self.mutator = DocumentMutator(cfg.mutator_args, llm=self.mutator_model)
        elif "sentence" in cfg.mutator_args.type:
            self.mutator_model = get_or_create_model(cfg.mutator_args.model_id, cfg.mutator_args)
            self.mutator = SentenceMutator(cfg.mutator_args, llm=self.mutator_model)
        elif "span" in cfg.mutator_args.type:
            self.mutator = SpanFillMutator()
        elif "word" in cfg.mutator_args.type:
            self.mutator = MaskFillMutator()
        else:
            raise ValueError("Invalid mutator type. Choose 'word', 'span', 'sentence' or 'document'.")

    def attack(self, prompt, watermarked_text):
        current_text = watermarked_text
        original = watermarked_text

        # Preliminary check to ensure that there is some watermark to attack
        watermark_detected, score = self.watermarker.detect(original)
        if not watermark_detected:
            raise ValueError("No watermark detected on input text. Nothing to attack! Exiting...")

        # Attack loop       
        backtrack_patience =  0
        results, mutated_texts = [], [original]
        for step_num in tqdm(range(self.cfg.attack_args.max_steps)):

            step_data = self.base_step_metadata
            step_data.update({"step_num": step_num})
            step_data.update({"current_text": current_text})

            log.info(f"step_data: {step_data}")

            # Potentially discard the current step and retry a previous one
            backtrack = backtrack_patience > self.cfg.attack_args.backtrack_patience
            if backtrack:
                log.error(f"Backtrack patience exceeded. Reverting mutated text to previous version.")
                backtrack_patience = 0
                if len(mutated_texts) > 1:
                    del mutated_texts[-1]
                    watermarked_text = mutated_texts[-1]

            # Step 1: Mutate      
            log.info("Mutating watermarked text...")
            mutated_text = self.mutator.mutate(current_text)
            # TODO: This should be cleaner. This is here to run the attack right now before I go to the gym. - Boran.
            if self.cfg.mutator_args.type == "sentence":
                mutated_text = mutated_text["mutated_text"]
            step_data.update({"mutated_text": mutated_text})
            if self.cfg.attack_args.verbose:
                log.info(f"Mutated text: {mutated_text}")

            # Step 2: Length Check
            log.info(f"Checking mutated text length to ensure it is within {self.cfg.attack_args.length_variance*100}% of the original...")
            length_issue, original_len, mutated_len = length_diff_exceeds_percentage(
                text1=original, 
                text2=mutated_text, 
                percentage=self.cfg.attack_args.length_variance
            )
            current_text_len = count_num_of_words(current_text)
            step_data.update({"current_text_len": current_text_len})
            step_data.update({"mutated_text_len": mutated_len})
            step_data.update({"length_issue": length_issue})

            if length_issue:
                log.warn(f"Failed length check. Previous was {original_len} words and mutated is {mutated_len} words. Skipping quality check and watermark check...")
                backtrack_patience =+ 1
                results.append(step_data)
                save_to_csv([step_data], self.cfg.attack_args.log_csv_path) 
                continue

            log.info("Length check passed!")

            # Step 3: Check Quality
            log.info("Checking quality oracle...")
            quality_analysis = self.quality_oracle.is_quality_preserved(prompt, current_text, mutated_text)
            step_data.update({"quality_analysis": quality_analysis})
            step_data.update({"quality_preserved": quality_analysis["quality_preserved"]})
        
            if not quality_analysis["quality_preserved"]:
                log.warn("Failed quality check. Skipping watermark check...")
                results.append(step_data)
                save_to_csv([step_data], self.cfg.attack_args.log_csv_path) 
                continue
            
            # If we reach here, that means the quality check passed, so update the current_text.
            current_text = mutated_text

            log.info("Quality check passed!")
            mutated_texts.append(mutated_text)

            # Step 4: Check Watermark
            watermark_detected, watermark_score = self.watermarker.detect(mutated_text)
            step_data.update({"watermark_detected": watermark_detected})
            step_data.update({"watermark_score": watermark_score})
            results.append(step_data)
            save_to_csv([step_data], self.cfg.attack_args.log_csv_path) 

            if not watermark_detected:
                log.info("Attack successful!")
                return mutated_text
            log.info("Watermark still present, continuing on to another step!")

        return original

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    
    import os

    CUDA_VISIBLE_DEVICES = str(cfg.cuda_visible_devices)
    WORLD_SIZE = str(len(str(cfg.cuda_visible_devices).split(",")))

    print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}")
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["WORLD_SIZE"] = WORLD_SIZE

    prompt = """
    "Summarize the plot and key details of the movie Captain Marvel, including its setting in the 1990s, main characters, and important scenes from the trailer. Mention the directors, writers, cast, and any interesting production facts or fan theories related to the film."
    """

    watermarked_text = """
    "Set in the 1990s, Captain Marvel follows Danvers, an Air Force officer, as she becomes one of the universe's most powerful heroes after earth gets involved in a galactic war between alien. There is no doubt that there are many superhero movies out there but it has been.
    Captain marvel and the kree-skrull war explained:
    Marvel Studios released the first trailer for Captain Marvel on September 18, 2018, and it gave us our first look at Brie Larson as Carol Danvers/Captain Marvel. However, the trailer didn't delve too deeply into the plot or the characters. But now, thanks to some new promo art and details from the film's stars and director, we have a better understanding of what's going on in the movie.

    Captain marvel trailer breakdown:

    The Captain Marvel trailer shows us a mix of action, humor, and emotional moments, giving us a glimpse into the life of Carol Danvers (Brie Larson), aka Captain Marvel. Here's a breakdown of what we see in the trailer:

        * The opening shot shows Carol crashing into a Blockbuster Video store in the '90s, which gives away the time period of the film.
        * We then cut to Carol in her suit, looking fierce and ready for battle.
        * Next, we see Nick Fury (Samuel L. Jackson) without his iconic eye patch, which suggests this might be before he lost his eye.
        * The trailer cuts to a montage of action sequences, showing Carol fighting Kree warriors and destroying enemy ships.",semstamp_lsh,MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ
    2,"Set in the 1990s, Captain Marvel follows Danvers, an Air Force officer, as she becomes one of the universe's most powerful heroes after earth gets involved in a galactic war between alien. The movie is set to release on March 8, 2019.
    Directed by: Anna Boden and Ryan Fleck Written by: Geneva Robertson-Dworet, Nicole Perlman, Meg LeFauve Cast: Brie Larson, Samuel L. Jackson, Ben Mendelsohn, Djimon Hounsou, Lee Pace, Lashana Lynch, Gemma Chan, Annette Bening, Clark Gregg Music by: Pinar Toprak Produced by: Marvel Studios Distributed by: Walt Disney Studios Motion Pictures #CaptainMarvel #BrieLarson #MarvelStudios #AvengersEndgame #MarvelCinematicUniverse #MCU Follow me on Instagram for more updates - instagram.com/the_movie_buff/ Facebook - facebook.com/themoviebuff/
    What if Captain Marvel was introduced earlier in the MCU? My thought process:
    Subscribe Here! 🤩👉 bit.ly/2CQq3oP Join The Discord Server! discord.gg/SyHrD6K Become A Patron! patreon.com/TheFilmTheorist Support Me On Ko-Fi! ko-fi.com/thefilmtheorist #CaptainMarvel #MCU #MarvelStudios #TheFilmTheorist **Copyright Disclaimer Under Section 107 of the Copyright Act 1976, allowance is made for ""fair use"" for purposes such as criticism, comment, news reporting, teaching, scholarship, and research.",semstamp_lsh,MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ
    3,"Set in the 1990s, Captain Marvel follows Danvers, an Air Force officer, as she becomes one of the universe's most powerful heroes after earth gets involved in a galactic war between alien. Here are some fun facts about the movie.

    **1. The Movie Was Written and Directed by Anna Boden and Ryan Fleck**: Boden is the first female director to helm a solo superhero film in the Marvel Cinematic Universe (MCU). She previously directed It's Kind of a Funny Story (2010) and Mississippi Grind (2015).

    **2. Brie Larson Trained for Nine Months to Prepare for the Role**: The actress underwent intense physical training, including martial arts, stunt training, and weightlifting, to play Carol Danvers/Captain Marvel. Her hard work paid off – she can do most of her own stunts!

    **3. The Film Takes Inspiration from the Comics**: The story draws inspiration from Kelly Sue DeConnick's comic book run, particularly the 2014 series Captain Marvel Vol. 7: Earth's Mightiest Hero. Fans of the comics will recognize nods to iconic characters, like Goose the cat.

    **4. Skrulls Were Created Using Advanced VFX Technology**: To bring the shapeshifting aliens to life, visual effects company Digital Domain developed cutting-edge technology to create realistic facial expressions and movements. No prosthetics or makeup were used for the Skrull designs.

    **5. Samuel L. Jackson's Nick Fury Has a Few Surprises**: In this '90s-set film, Nick Fury is a younger, more optimistic agent."
    """
    
    attacker = Attack(cfg)
    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
