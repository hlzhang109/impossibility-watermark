import os
from openai import OpenAI
import random
from utils import find_csv, get_mutated_text, query_openai_with_history

# Set your OpenAI API key here
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) # load openai api key from ./.env

def distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2):
    print("---------------------------------------------------------------")

    initial_prompt = f"""Here's a prompt and two possible responses. Compare the responses and explain the difference between them.

Prompt

{prompt}

Response 1

{response_1}

Response 2

{response_2}
    """
    
    if random.choice([True, False]):
        perturbed_1, perturbed_2 = perturbed_2, perturbed_1
        print("Flipped.")
    
    
    follow_up_prompt = f"""Here are two completely different responses. See if you can match these responses to Response 1 and Response 2. At the end of your response, output 1 if you match Response A with Response 1 and 2 if you match response A with response 2.

Response A
{perturbed_1}

Response B

{perturbed_2}"""
    
    first_response, second_response = query_gpt4_with_history(initial_prompt, follow_up_prompt)
    print("First Response:", first_response)
    print("Second Response:", second_response)
    print("---------------------------------------------------------------")
    
def main():
    # First Prompt 
    prompt = "Write a 250-word short story about a married American tourist called Evan falling in love with a barista called Emily at a Parisian cafe. Only respond with the story."
    
    response_1 = """Evan, an American tourist, found himself wandering the cobblestone streets of Paris in the spring, the city's charm enveloping him like a warm embrace. On a quiet corner, he stumbled upon a quaint café, its aroma of freshly brewed coffee and pastries luring him inside. There, behind the counter, was Emily, a barista whose smile was as inviting as the café itself.\nEach morning, Evan returned, drawn not only by the coffee but by the light in Emily's eyes. They exchanged stories and laughter, the language barrier dissolving between sips of espresso and shared croissants. Evan, though married, felt an unexpected connection, a stirring in his heart that he couldn't ignore.\nParis, with its timeless beauty, became the backdrop for Evan's internal turmoil. He was torn between his commitment back home and the unexpected feelings for Emily. Their brief encounters in the café blossomed into walks along the Seine, each moment together etching a deeper mark on Evan's heart.\nAs his trip neared its end, Evan realized that Paris had offered him not just a journey through its streets but through the depths of his own heart. On his last visit to the café, he confessed his feelings to Emily. She listened, her eyes reflecting a mix of sorrow and understanding. They knew their connection was fleeting, a beautiful but brief encounter.\nEvan returned home, his heart heavy yet awakened. Paris, and Emily, had changed him, teaching him that love can be found in the most unexpected places, even if it's not meant to last forever."""
    response_2 = """In the heart of Paris, where cobblestone streets whispered tales of love, Evan, an American tourist, found himself wandering aimlessly, captivated by the city's charm. Married yet alone, he sought solace in a quaint Parisian café, a refuge from his solitary explorations.\nThe café, a mosaic of Parisian elegance and coziness, greeted him with the rich aroma of freshly brewed coffee. There, behind the counter, was Emily, a barista whose smile seemed to blend seamlessly with the café's warm ambiance. Her eyes, a deep shade of hazel, sparkled with a kind of magic unique to those who truly loved their city.\nEvan, initially just another customer, found himself returning daily, drawn not just by the allure of coffee but by the enchanting presence of Emily. Their conversations, initially revolving around trivial café recommendations, gradually delved deeper, meandering through dreams, aspirations, and laughter.\nIn Emily's company, Evan rediscovered a part of himself long forgotten. She listened with an earnestness that made him feel seen, understood. It was a connection he hadn't realized he was missing, a feeling of being genuinely alive.\nYet, amidst this unexpected whirlwind of emotions, Evan was tethered to reality by the ring on his finger—a symbol of commitment, a reminder of the life he had built back home. His heart, caught in a bittersweet symphony, grappled with the complexities of love and loyalty.\nAs his trip neared its end, Evan sat at the café for one last time. Words remained unspoken, yet their eyes conveyed volumes. In that moment, they both understood that what they shared was a fleeting chapter in life's grand narrative, a beautiful yet transient connection.\nEvan left Paris with a heart full of memories and a soul touched by a serendipitous encounter. Emily, the Parisian barista, remained a sweet, ephemeral dream, a testament to the unpredictable journeys of the heart."""
    
    # Second Prompt
    prompt = "Write a 250-word short story about a married American tourist called Evan falling in love with a barista called Emily at a Parisian cafe during spring. Evan and Emily should go for walks along the Seine. Evan should talk to Emily about his feelings before leaving. Only respond with the story."
    
    # Third prompt
    # prompt = """Write a 250-word story about Evan, a married American tourist, who falls for Emilie, a lively barista, during a spring festival in Paris. They bond over their love for art and the city's charm, leading to walks along the Seine, where the festival's joy spills over. Their connection deepens amid shared laughter and explorations of quaint streets. As the festival lights dance on the river, Evan shares his feelings with Emilie under the starlit sky, promising to cherish the moments they've shared. Focus on their joyful adventures and the sincere emotions that bloom between them."""
    # response_1 = """Evan, an American tourist, found himself captivated by the vibrant spirit of Paris during a spring festival. Amidst the lively streets adorned with blossoms and the air filled with joyous melodies, he stumbled upon a quaint café where Emilie, a barista with an infectious smile, brewed magic in a cup. Their initial exchange over a love for art and the city's undying charm quickly blossomed into a series of adventures along the Seine, where the festival's exuberance seemed to mirror their growing affection.\nTheir walks were filled with shared laughter and endless conversations, as they explored the city's quaint streets and hidden gems. Paris, in its festival attire, provided the perfect backdrop for their budding connection, with its cobbled lanes and the Seine's banks lit by the festival lights, dancing on the water like fireflies under the starlit sky.\nOne evening, as they found themselves under the canopy of stars, with the city's lights reflecting in their eyes, Evan shared his feelings with Emilie. The joy and sincerity in his voice resonated with the genuine emotions that had taken root in their hearts. He promised to cherish the moments they had shared, acknowledging the special bond they had formed. In the heart of Paris, amidst the spring festival's enchantment, Evan and Emilie discovered a connection that transcended the ordinary, marked by sincere emotions and joyful adventures, forever etched in their memories."""
    # response_2 = """Evan, an American tourist, found himself captivated not just by the allure of Paris in spring but by Emilie, a lively barista with a smile that mirrored the city's charm. Their connection sparked during a bustling spring festival, where art and joy intertwined, drawing them closer over shared passions.\n\nTheir days were filled with wanderings along the Seine, where the festival's exuberance seemed to overflow, blending seamlessly with the serene flow of the river. Laughter became their shared language as they explored the city's quaint streets and hidden gems, with Emilie introducing Evan to the true essence of Paris beyond the postcards.\n\nArt, a mutual love, allowed them to see the world through each other's eyes. They lost themselves in galleries and street art, finding pieces of themselves in the colors and shapes. The city, with its endless charm, became the backdrop for their growing connection, a testament to the unexpected paths of the heart.\n\nAs the festival lights danced on the river's surface, Evan found the courage under the starlit sky to share his feelings with Emilie. Amid the magic of the moment, he spoke of the joy she had brought into his life, promising to cherish the memories they had created together. Their story, though fleeting, was a vivid tapestry of laughter, exploration, and sincere emotions, a reminder of the beauty found in shared experiences and the unpredictable journey of the heart."""
    
    # Get Perturbed Versions
    csv_file_directory = "./eval/results/"
    first_perturbed_csv_filename = "1_evan_1_2_1.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    
    perturbed_1 = get_mutated_text(csv_file_path)
    
    second_perturbed_csv_filename = "1_evan_2_1_1.csv"
    csv_file_path = os.path.join(csv_file_directory, second_perturbed_csv_filename)
    
    perturbed_2 = get_mutated_text(csv_file_path)
    
    
    print(f"Prompt: {prompt}")
    print(f"Response 1: {response_1}")
    print(f"Response 2: {response_2}")
    print(f"Perturbed 1: {first_perturbed_csv_filename}")
    print(f"Perturbed 2: {second_perturbed_csv_filename}")
    
    for _ in range(10):
        distinguish(prompt, response_1, response_2, perturbed_1, perturbed_2)

if __name__ == "__main__":
    main()