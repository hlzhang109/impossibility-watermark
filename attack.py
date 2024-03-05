import os

# os.environ["WORLD_SIZE"] = "1"

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
import re
import json
import pandas as pd

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
        self.watermarker  = Watermarker(cfg, pipeline=self.generator_pipe_builder, is_completion=cfg.attack_args.is_completion)
        self.quality_oracle = Oracle(cfg=cfg.oracle_args, pipeline=self.oracle_pipeline_builder.pipeline)
        self.mutator = TextMutator(cfg.mutator_args, pipeline=self.mutator_pipeline_builder.pipeline)

    def count_words(self, text):
        if text is None:
            return 0
        return len(text.split())

    def attack(self, cfg, prompt=None, watermarked_text=None):
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
        
        log.info(f"Original Watermarked Text: {original_watermarked_text}")
        
        # Log the original watermarked text
        perturbation_stats = [{
            "step_num": -1,
            "current_text": original_watermarked_text,
            "mutated_text": original_watermarked_text, 
            "current_text_len": self.count_words(original_watermarked_text),
            "mutated_text_len": self.count_words(original_watermarked_text), 
            "quality_preserved": True,
            "quality_analysis" : "No analysis.",
            "watermark_detected": watermark_detected,
            "watermark_score": score,
            "backtrack" : False,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }]
        save_to_csv(perturbation_stats, save_path)

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

            current_text_len = self.count_words(watermarked_text)
            mutated_text_len = self.count_words(mutated_text)

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
                    
            perturbation_stats = [{
                "step_num": step_num, 
                "current_text": watermarked_text,
                "mutated_text": mutated_text, 
                "current_text_len": self.count_words(watermarked_text),
                "mutated_text_len": self.count_words(mutated_text), 
                "quality_preserved": quality_preserved,
                "quality_analysis" : quality_analysis,
                "watermark_detected": watermark_detected,
                "watermark_score": score,
                "backtrack": backtrack,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }]
            save_to_csv(perturbation_stats, save_path)
            
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
    
def get_prompt_or_output(csv_path, num):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # Get the specific text based on num
    if num <= len(df) and num > 0:
        story = df.iloc[num - 1]['text']
    else:
        raise Exception(f"Index out of range.")
    
    return story

def find_csv(txt_file_path):
    with open(txt_file_path, 'r') as file:
        content = file.read()

    # Search for the first occurrence of 'attack_*.csv' pattern
    match = re.search(r'attack_.*\.csv', content)

    if match:
        csv_filename = match.group(0)
        return csv_filename
    return None

def get_mutated_text(txt_file_path):
    file_name = find_csv(txt_file_path)
    directory = "./eval/results/"
    file_path = os.path.join(directory, file_name)

    df = pd.read_csv(file_path)
    
    success_df = df[(df['mutated_text_len'] >= 0.95 * df['current_text_len']) & (df['quality_preserved'] == True)]

    return success_df['mutated_text'].iloc[-1]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.attack_args.cuda
    
    # Load prompt and watermarked text from JSON
    
    # file_path = "/home/borito1907/impossibility-watermark/text_completions_50_c4.json"
    # with open(file_path, 'r') as file:
    #     data = json.load(file)

    # # Initialize lists to store prefixes and completions
    # prefixes = []
    # completions = []

    # # Iterate through each element in the list
    # for item in data:
    #     prefixes.append(item['Prefix'])
    #     completions.append(item['Completion'])
    
    # index = 0
        
    # prompt = prefixes[index]
    # watermarked_text = completions[index]
    
    # Read the prompt and the watermarked text from the input files
    prompt = cfg.attack_args.prompt
    if prompt is None:
        prompt = get_prompt_or_output(cfg.attack_args.prompt_file, cfg.attack_args.prompt_num) 
        
    watermarked_text = cfg.attack_args.watermarked_text
    if watermarked_text is None and cfg.attack_args.watermarked_text_path is not None:
        watermarked_text = get_prompt_or_output(cfg.attack_args.watermarked_text_path, cfg.attack_args.watermarked_text_num)
    
    # watermarked_text = """In the heart of Paris, during the bloom of spring, Evan, an American tourist, found himself entranced by the city's magic. Married but restless, he sought solace in the quaint cafes lining the cobblestone streets. It was in one such cafe that he met Emily, a barista whose smile was as warm as the coffee she poured. Day after day, Evan returned, drawn not by the allure of caffeine but by the light in Emily's eyes.\nTheir conversations, initially casual, deepened like the river Seine that coursed through the city. Evan found himself sharing stories of his life back home, his dreams, and the growing void he felt. Emily, with her serene disposition and keen listening, became a balm to his weary soul.\nOne crisp evening, as they walked along the Seine, Evan confessed. Under the canopy of twinkling stars, he spoke of the unexpected affection he had for Emily, a feeling so profound it startled him. "I never intended to feel this way," he admitted, the city lights reflecting in his eyes. "But you've touched my heart in a way I can't ignore."\nEmily listened, her expression a mix of compassion and sorrow. She understood the weight of his confession, the turmoil it represented. As they stood by the river, the silence between them spoke volumes. Evan knew he had to leave Paris, to return to his life, but he also knew a part of his heart would forever remain with Emily, by the Seine, in the city of lights.\nAs Evan departed, the promise of spring lingered in the air, a testament to a love that was as unexpected as it was fleeting."""
    # watermarked_text = """During his visit to captivating Paris, an American tourist named Evan surrendered to the city's charm, finding tranquility in its bustling core. At a neighborhood fair, he discovered peace during the day, nestled in the metropolis' narrow, winding alleys, leading to cozy taverns echoing stories of yesteryears. A quaint café, managed by Emily, turned into a haven for him. Beyond providing nourishment, Emily's exceptional talent for showing appreciation to every customer created an atmosphere of fellowship and unity among the regulars. Her lively energy sparked Evan's interest, making their interactions valuable, similar to the rhythmic ebb and flow of the Seine River weaving through the heart of the city. Within this intimate setting, Evan disclosed his modest upbringing, revealed his aspirations, and confided his yearning for home. Moved by his story, Emily assisted in rekindling his motivation. As dusk fell, they walked along the serene banks of the Seine, engaged in quiet conversation under the gentle illumination of the full moon, enveloped in a soothing calm beneath the vast canopy of stars, reflecting their combined images. Here, Evan expressed his profound thoughts, revealing emotions he hadn't expected but couldn't deny. He admitted, 'I had envisioned a different scenario,' but he couldn't dismiss the strong bond between them. Emily acknowledged her growing affection too. His hesitancy allowed her to recognize her own uncertainties, drawing her nearer to him and acknowledging the profound importance of their bond. Under the brilliant moonlight, they considered the implications of his revelation. Facing his imminent departure, Evan pledged to leave a part of himself in Paris, specifically on the banks of the Seine, with Emily, maintaining the remarkable relationship - now an enduring memory."""
    # watermarked_text = """In the heart of Paris, as spring awakened the city with gentle warmth and the bloom of cherry blossoms, an American tourist named Evan found himself in a quaint café near the Seine. The aroma of freshly brewed coffee mingled with the scent of pastries, drawing him in. It was there he first saw Emily, a barista with a smile that seemed to outshine the Parisian sun.\nEach morning, Evan would visit the café, his heart drawn more to Emily's radiant presence than the allure of the coffee. They began to talk, sharing stories and laughter between orders. Emily's eyes sparkled with a passion for art and the beauty of her city, and Evan found himself captivated.\nAs days passed, their conversations spilled out of the café and onto the cobbled streets of Paris. They walked along the Seine, the river reflecting the golden hues of sunset, creating a canvas that mirrored the growing warmth between them. Evan listened, enraptured, as Emily spoke of her dreams, her words flowing like the gentle currents of the river.\nBut time, like the river, flowed relentlessly, and Evan's departure loomed near. On their final walk, under the soft glow of streetlights, Evan turned to Emily, his heart heavy. "Emily," he began, his voice a mix of sadness and sincerity, "I never expected to find someone like you. You've shown me a Paris that I never knew existed, one that I'll carry with me forever. I wish I could stay, or take you with me."\nEmily's eyes glistened, touched by his words. She reached for his hand, giving it a gentle squeeze. "Evan, you've brought joy to my days. Our time may have been short, but it was beautiful. Paris will always be here, waiting for you."\nAs Evan left Paris, he carried with him the memory of Emily and the Seine, a sweet reminiscence of springtime love."""
    
    # watermarked_text = """Evan, an American tourist married to his routine back home, found himself enchanted by more than just the spring festival's allure in Paris. Amidst the vibrant stalls and lively music that filled the air, it was Emilie, a vivacious barista with a laugh that echoed the city's joy, who captured his heart. Their initial encounter over a simple cup of coffee blossomed into a series of shared adventures, as they discovered a mutual love for art and the irresistible charm of Paris.\nAs the festival breathed life into the city, Evan and Emilie found themselves wandering along the Seine, where the festival's jubilance seemed to spill over into their every step. Their connection deepened with each shared laugh and every quaint street they explored together. Paris, with its timeless beauty, became the backdrop to their burgeoning bond, a testament to the moments they were weaving together, thread by thread.\nThe nights brought a magical transformation, with the festival lights dancing on the river's surface, mirroring the sparkle in their eyes. It was under one such starlit sky, with the gentle hum of the city around them, that Evan shared his feelings with Emilie. Though aware of the complexities of his life, he spoke of a promise not to bind her to promises of forever but to cherish the sincerity and joy of the moments they shared.\nTheir story was a testament to the unexpected connections that life can bring, marked by joyful adventures and the sincere emotions that bloom in the most fleeting of moments."""
    # watermarked_text = """"Evan, an advocate for 'flânerie,' eagerly sought the excitement of a spring festival in Paris, savoring its thrill and profound symbolic significance. Upon entering the lively marketplace, throbbing with festive music, he was quickly drawn to Émilie, the captivating barista, whose radiant energy mirrored the vibrant essence of the city. During their initial encounter over casual drinks, they delved into various shared memories, strengthening their relationship and revealing their mutual admiration for modern art and the allure of Paris. Walking lazily along the Seine, they basked in the jubilant ambiance, discovering inspiration and tranquility within the vivid urban landscape. Deeper into their exploration, they stumbled upon concealed gems, forming lasting memories and solidifying their connection. At the core of Paris, her feelings became entwined with the city's intricate tapestry, weaving a distinctive love story distinct from any other. As dusk settled and the city's reflection shimmered on the river's surface, a pivotal choice came to light, surpassing temporal bounds. Beneath the starlit sky, amidst the soft whispers of Paris, Evan articulated his deepest sentiments to Émilie, pledging to uphold and cherish the genuineness of their relationship as they embarked on their transient voyage together. Their escapade embodied life itself, powered by fervent dialogues and fervent affection, driven by evolutionary forces."""
    # watermarked_text = """Evan, an American tourist, found himself captivated by the vibrant spirit of Paris during a spring festival. Amidst the lively streets adorned with blossoms and the air filled with joyous melodies, he stumbled upon a quaint café where Emilie, a barista with an infectious smile, brewed magic in a cup. Their initial exchange over a love for art and the city's undying charm quickly blossomed into a series of adventures along the Seine, where the festival's exuberance seemed to mirror their growing affection.\nTheir walks were filled with shared laughter and endless conversations, as they explored the city's quaint streets and hidden gems. Paris, in its festival attire, provided the perfect backdrop for their budding connection, with its cobbled lanes and the Seine's banks lit by the festival lights, dancing on the water like fireflies under the starlit sky.\nOne evening, as they found themselves under the canopy of stars, with the city's lights reflecting in their eyes, Evan shared his feelings with Emilie. The joy and sincerity in his voice resonated with the genuine emotions that had taken root in their hearts. He promised to cherish the moments they had shared, acknowledging the special bond they had formed. In the heart of Paris, amidst the spring festival's enchantment, Evan and Emilie discovered a connection that transcended the ordinary, marked by sincere emotions and joyful adventures, forever etched in their memories."""
    # watermarked_text = """Evan, an American tourist, found himself captivated not just by the allure of Paris in spring but by Emilie, a lively barista with a smile that mirrored the city's charm. Their connection sparked during a bustling spring festival, where art and joy intertwined, drawing them closer over shared passions.\n\nTheir days were filled with wanderings along the Seine, where the festival's exuberance seemed to overflow, blending seamlessly with the serene flow of the river. Laughter became their shared language as they explored the city's quaint streets and hidden gems, with Emilie introducing Evan to the true essence of Paris beyond the postcards.\n\nArt, a mutual love, allowed them to see the world through each other's eyes. They lost themselves in galleries and street art, finding pieces of themselves in the colors and shapes. The city, with its endless charm, became the backdrop for their growing connection, a testament to the unexpected paths of the heart.\n\nAs the festival lights danced on the river's surface, Evan found the courage under the starlit sky to share his feelings with Emilie. Amid the magic of the moment, he spoke of the joy she had brought into his life, promising to cherish the memories they had created together. Their story, though fleeting, was a vivid tapestry of laughter, exploration, and sincere emotions, a reminder of the beauty found in shared experiences and the unpredictable journey of the heart."""
    
    # TODO: TWO WATERMARKED TEXTS TO RUN
    # watermarked_text = """Evan, an American tourist with a keen eye for beauty and a heart open to adventures, found himself wandering the vibrant streets of Paris during the enchanting spring festival. Amidst the jubilant chaos of colors and sounds, his path crossed with Emilie's, a lively barista whose smile was as inviting as the city itself. Their first encounter was a symphony of laughter and playful banter over a shared love for Impressionist art and the timeless charm of Paris.\nAs the festival's joy bubbled around them, Evan and Emilie found themselves drawn together, strolling along the Seine. The river mirrored the festival's lights, creating a mesmerizing dance of colors that seemed to celebrate their newfound connection. Their walks became a canvas for their burgeoning relationship, painted with shared stories, laughter, and a mutual fascination with the quaint, cobblestone streets that whispered the city's secrets.\nThe days flew by, each moment more precious than the last, as they explored Paris's hidden gems, from cozy cafes to tucked-away art galleries. Amidst the beauty of the city in bloom and the warmth of the festival's embrace, their bond deepened, rooted in genuine affection and mutual understanding.\nUnder a starlit sky, with the festival's magic lingering in the air, Evan found the courage to share his feelings with Emilie. Amid the serene glow of the river, he promised to cherish the memories they'd created, the laughter they'd shared, and the connection that had blossomed. Their story was a testament to the joy of unexpected encounters and the beauty of letting your heart lead the way in the city of love."""
    watermarked_text = """In the heart of Paris, as the city blooms into the vibrant colors of spring, Evan, an American tourist with a deep-seated love for art and culture, finds himself enchanted by the city’s spring festival. Amid the cobblestone streets adorned with festoons and the air filled with melodies, his path crosses with Emilie, a lively Parisian barista with a smile that could rival the city’s enchanting lights.\nTheir connection sparks over steaming cups of coffee and tales of their favorite artists, quickly blossoming as they explore the festival together. They wander along the Seine, where the festival's joy overflows, blending with the laughter of children and the tunes of street musicians. As they meander through quaint streets, discovering hidden gems of the city, their conversations flow as smoothly as the river beside them, filled with dreams, passions, and shared laughter.\nWith each day, their adventures through Paris bring them closer, from marveling at masterpieces in the Louvre to enjoying crepes under the shade of blooming chestnut trees. The festival lights, reflecting off the Seine, provide a backdrop to their growing affection, mirroring the spark in their eyes.\nOne starlit night, by the riverbank, amidst the gentle hum of the city, Evan finds the courage to share his feelings with Emilie. Under the canopy of stars, he speaks of the precious moments they’ve shared, promising to cherish this unexpected connection. Their story, a testament to the magic of Paris in spring, remains a celebration of joy, art, and the serendipitous love that bloomed between them."""
    
    # txt_file_directory = "./first_round/"
    # txt_file_name = "evan_1_4.txt"
    # txt_file_path = os.path.join(txt_file_directory, txt_file_name)
    
    # watermarked_text = get_mutated_text(txt_file_path)
    
    attacker = Attack(cfg)
    attacked_text = attacker.attack(cfg, prompt, watermarked_text)
                
    log.info(f"Prompt: {prompt}")
    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
        