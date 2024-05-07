import os
import datetime
import textwrap
import traceback

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from model_builders.pipeline import PipeLineBuilder 

import logging
import hydra

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class DistinguisherAnswer(BaseModel):
    answer: int = Field(description="Which story did the LLM match Story A to?")

class Distinguisher:
    def __init__(self, cfg, pipeline=None) -> None:        
        self.cfg = cfg # config.oracle_args
        self.pipeline = pipeline

        # Prompt Template
        self.profile_background = """{system_profile}"""

        self.first_instructions = textwrap.dedent("""
Story 1: {{story_1}}

Story 2: {{story_2}}

Story A: {{story_a}}
                                            
Story A was created by modifying Story 1 or Story 2. Which one? Explain your reasoning in-depth before responding. Both stories have similar themes and plots, so focus on specific details to
make a decision.""")

        self.first_prompt = PromptTemplate(
            template=self.first_instructions,
            input_variables=["story_1", "story_2", "story_a"],
        )

        # Model Pipeline
        if not isinstance(self.pipeline, PipeLineBuilder):
            log.info("Initializing a new Distinguisher pipeline from cfg...")
            self.pipeline = PipeLineBuilder(self.cfg)

        self.first_chain = self.first_prompt | self.pipeline

        self.second_instructions = textwrap.dedent("""
Response: {{response}}
                                                   
Read the above the response given by another LLM. Does the LLM think Story A was created from Story 1 or Story 2?
                                                   
If the LLM thinks it is Story 1, respond with a 1. If the LLM thinks it is story 2, respond with a 2.""")
        
        # self.second_output_parser = PydanticOutputParser(pydantic_object=DistinguisherAnswer)
        
        self.second_prompt = PromptTemplate(
            template=self.second_instructions,
            input_variables=["response"],
        )

        self.second_chain = self.second_prompt | self.pipeline

    def match(self, story_1, story_2, story_a, **kwargs):
        # Prepare Input
        dict_input = {
            "story_1": story_1, 
            "story_2": story_2,
            "story_a": story_a,
        }

        # Run Chain
        response = self.first_chain.invoke(dict_input)

        log.info(f"Analysis: {response}")

        dict_input = {
            "response": response,
        }

        output = self.second_chain.invoke(dict_input)

        log.info(f"Final Output: {output}")

        return output


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg):

    response_1 = textwrap.dedent("""Evan, an American tourist, found himself wandering the cobblestone streets of Paris in the spring, the city's charm enveloping him like a warm embrace. On a quiet corner, he stumbled upon a quaint café, its aroma of freshly brewed coffee and pastries luring him inside. There, behind the counter, was Emily, a barista whose smile was as inviting as the café itself.
    Each morning, Evan returned, drawn not only by the coffee but by the light in Emily's eyes. They exchanged stories and laughter, the language barrier dissolving between sips of espresso and shared croissants. Evan, though married, felt an unexpected connection, a stirring in his heart that he couldn't ignore.
    Paris, with its timeless beauty, became the backdrop for Evan's internal turmoil. He was torn between his commitment back home and the unexpected feelings for Emily. Their brief encounters in the café blossomed into walks along the Seine, each moment together etching a deeper mark on Evan's heart.
    As his trip neared its end, Evan realized that Paris had offered him not just a journey through its streets but through the depths of his own heart. On his last visit to the café, he confessed his feelings to Emily. She listened, her eyes reflecting a mix of sorrow and understanding. They knew their connection was fleeting, a beautiful but brief encounter.
    Evan returned home, his heart heavy yet awakened. Paris, and Emily, had changed him, teaching him that love can be found in the most unexpected places, even if it's not meant to last forever.
    """)

    response_2 = textwrap.dedent("""In the heart of Paris, where cobblestone streets whispered tales of love, Evan, an American tourist, found himself wandering aimlessly, captivated by the city's charm. Married yet alone, he sought solace in a quaint Parisian café, a refuge from his solitary explorations.
    The café, a mosaic of Parisian elegance and coziness, greeted him with the rich aroma of freshly brewed coffee. There, behind the counter, was Emily, a barista whose smile seemed to blend seamlessly with the café's warm ambiance. Her eyes, a deep shade of hazel, sparkled with a kind of magic unique to those who truly loved their city.
    Evan, initially just another customer, found himself returning daily, drawn not just by the allure of coffee but by the enchanting presence of Emily. Their conversations, initially revolving around trivial café recommendations, gradually delved deeper, meandering through dreams, aspirations, and laughter.
    In Emily's company, Evan rediscovered a part of himself long forgotten. She listened with an earnestness that made him feel seen, understood. It was a connection he hadn't realized he was missing, a feeling of being genuinely alive.
    Yet, amidst this unexpected whirlwind of emotions, Evan was tethered to reality by the ring on his finger—a symbol of commitment, a reminder of the life he had built back home. His heart, caught in a bittersweet symphony, grappled with the complexities of love and loyalty.
    As his trip neared its end, Evan sat at the café for one last time. Words remained unspoken, yet their eyes conveyed volumes. In that moment, they both understood that what they shared was a fleeting chapter in life's grand narrative, a beautiful yet transient connection.
    Evan left Paris with a heart full of memories and a soul touched by a serendipitous encounter. Emily, the Parisian barista, remained a sweet, ephemeral dream, a testament to the unpredictable journeys of the heart.
    """)

    response_1_perturbed_first = """Evan, a North American traveler, happened upon the cobblestone alleys of Paris in the springtime, the city's allure wrapping around him like a cozy blanket. At a serene intersection, he came across a small cafÃ©, its scent of newly brewed coffee and baked goods pulling him inward. Here, stationed behind the counter, was Emily, a barista whose grin was as welcoming as the cafÃ©. Every morning, Evan made his way back, enticed not solely by the coffee but also by the gleam in Emily's eyes. They shared tales and amusement, the language barrier softening over cups of espresso and communal croissants. Despite being a married individual, Evan experienced an unfamiliar bond, a fluttering in his heart that he could not overlook. Paris, with its eternal elegance, turned into the setting for Evan's emotional conflict. Torn between his pledge at home and the novel sentiments for Emily, his days in Paris were filled with exploration externally and internally. These blossoming encounters in the cafÃ© grew into strolls alongside the Seine, every second together carving a deeper imprint on Evan's spirit. Approaching the conclusion of his trip, Evan acknowledged that Paris presented him not merely a voyage through its streets but also through the complexities of his own soul. Upon his final visit to the cafÃ©, he revealed his emotions to Emily. Her gaze reflected a mix of sympathy and understanding as she graciously accepted his words. Both understood that their bond was transient, a lovely yet ephemeral affair. Evan went back home, his heart burdened yet illuminated. Paris, and Emily, left a lasting impression on him, confirming that affection can emerge in the most unforeseen circumstances, even if it might not endure everlastingly."""
    response_1_perturbed_last  = """Evan, a North American traveler, arrived in Paris during the spring. He was captivated by the city's charm, which enveloped him like a warm blanket. In a quiet corner, he discovered a quaint cafÃ©. Its aroma of fresh coffee and pastries drew him in. Behind the counter, Emily, a barista, greeted him with a smile as inviting as the cafÃ© itself. Each morning, Evan returned, attracted not only by the coffee but also by Emily's radiant smile. They exchanged stories and laughter, the language barrier softening over shared espressos and morning croissants. Despite being married, Evan felt an unexpected connection, a flutter in his chest he couldn't ignore. His emotional journey unfolded in the timelessly beautiful Paris. Torn between his commitment at home and his newfound feelings for Emily, his days in Paris were filled with exploration and internal conflict. These blossoming interactions at the cafÃ© led to walks along the Seine, each moment together etching a deeper mark on Evan's spirit. As his journey neared its end, he acknowledged that Paris had given him more than just a sightseeing experience; it had taken him on a path of self-discovery and understanding. On his final visit to the cafÃ©, he revealed his feelings to Emily. She looked into his eyes, reflecting empathy and understanding as she kindly received his words. 'Their bond was transient,' they acknowledged, recognizing its ephemeral beauty. Evan returned home, his heart heavy yet enlightened. Paris, and Emily, had left an indelible impression on him, affirming that love can emerge in the most unlikely circumstances, even if it may not last forever."""

    response_2_perturbed_first = """In the heart of Paris, where cobblestone streets murmured stories of love, Evan, an American tourist, strolled leisurely, captivated by the city's charm. Married but alone, he sought comfort in a charming Parisian cafÃ©, a haven from his solitary expeditions. This cafÃ©, a harmonious blend of Parisian sophistication and warmth, welcomed him with the tantalizing aroma of newly brewed coffee. Here, behind the counter, worked Emily, a barista whose smile appeared to meld with the cafÃ©'s inviting atmosphere. Her eyes, a rich hue of hazel, shimmered with a distinct enchantment unique to those who deeply cherish their city. Initially, Evan was merely another patron; however, he became a regular visitor, enticed not only by the appeal of the coffee but also by the bewitching presence of Emily. At first, their exchanges were about trivial cafÃ© suggestions, but soon enough, they traversed deeper territories, exploring dreams, hopes, and laughter. Through Emily, Evan rediscovered a dormant fragment of his persona. Her attentiveness was so profound that he felt acknowledged, comprehended. It was a bond he had not recognized he yearned for, a sensation of authentic vitality. However, amidst this emotional tempest, Evan was grounded by the band on his finger - a pledge, a recollection of the life he had constructed back home. His feelings, a poignant symphony, wrestled with the intricacies of affection and duty. As his journey drew to a close, Evan visited the cafÃ© for a final time. No words were spoken, but their gazes communicated more than any dialogue could. They both understood that their relationship was a fleeting episode in life's sweeping saga, a precious yet transient attachment. Upon departing Paris, Evan carried with him a treasure trove of memories and a spirit stirred by a fateful encounter. Emily, the Parisian barista, remained a fond, transient memory, a testament to life's unforeseen adventures."""
    response_2_perturbed_last  = """As Evan, an American tourist, strolled leisurely through the charming heart of Paris, captivated by whispers of love from its narrow alleyways, he felt mysteriously drawn to a traditional Parisian cafÃ©. This establishment merged French elegance with cozy warmth. Approaching it, the enticing aroma of freshly brewed coffee filled the air, an alluring representative of the city's soul. Emily, the barista, welcomed patrons with a radiant and cheerful smile, adding a comforting touch to the cafÃ©'s inviting ambiance. Her charisma attracted both locals and foreigners, shining like stars in the night sky as she expertly crafted beverages. Initially, Evan was merely another customer. However, frequent visits spurred by the cafÃ©'s charm and Emily's engaging demeanor led him to become a regular. Their conversations evolved from trivialities to deeper topics, such as dreams, aspirations, and shared senses of humor. Emily helped Evan rediscover a long-forgotten side of himself, something he hadn't experienced in quite some time - undivided attention during their meetings. Though deeply connected to his past, Evan remained anchored, bound by the wedding ring adorning his finger, symbolizing a sacred vow and a constant reminder of the life he had constructed before embarking on this new journey. He navigated the complex emotions in his marriage, resonating with the harmonious melody of a symphony as he explored a rich tapestry of feelings, all while maintaining his commitment. As he prepared to bid farewell, Evan visited the cafÃ© one last time, tearfully saying goodbye. A poignant silence settled between them, communicating more than words ever could. Aware that their relationship represented a transient moment within the larger scope of life, they cherished the profound bond formed, a brief yet meaningful connection etched in the grand narrative of existence. With a heart full of gratitude, Evan departed from Paris, enriched in spirit and abundant with memories, forever indebted to an unexpected encounter that had breathed new life into his world. Emily, the Parisian barista, would perpetually occupy a special place in his heart, an eternal embodiment of happiness."""

    distinguisher = Distinguisher(cfg.generator_args)

    evaluation_1_3_first = distinguisher.match(response_1, response_2, response_1_perturbed_first)
    evaluation_1_3_last = distinguisher.match(response_1, response_2, response_1_perturbed_last)
    print(f"evaluation_1_3_first: {evaluation_1_3_first}")
    print(f"evaluation_1_3_last: {evaluation_1_3_last}")

    evaluation_2_3_first = distinguisher.match(response_1, response_2, response_2_perturbed_first)
    evaluation_2_3_last = distinguisher.match(response_1, response_2, response_2_perturbed_last)
    print(f"evaluation_2_3_first: {evaluation_2_3_first}")
    print(f"evaluation_2_3_last: {evaluation_2_3_last}")

if __name__ == "__main__":
    test()

