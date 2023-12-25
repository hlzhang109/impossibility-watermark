
import time
import torch
import transformers
import re
import math
import jsonlines
from cmd_args import get_cmd_args
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from oracle import *

import json

def append_dict_to_json_file(file_path, new_data):
    """
    Append a dictionary to a JSON file. Assumes the file contains an array of objects.

    :param file_path: Path to the JSON file
    :param new_data: Dictionary to append
    """
    try:
        with open(file_path, 'r') as file:
            # Load the existing data
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new list
        data = []

    # Append the new dictionary to the list
    data.append(new_data)

    # Write the modified list back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_intermediate_results(data, filename):
    df_out = pd.DataFrame(data)
    if os.path.exists(filename):
        df_out.to_csv(filename, mode='a', header=False, index=False)  # Append without writing headers
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_out.to_csv(filename, index=False)  # Create new file with headers
    print(f"Data appended to {filename}")


class Attacker:
    def __init__(self) -> None:
        self.n_resample = 5
        self.args = get_cmd_args()
        self.mask_filling_model_name = self.args.mask_filling_model_name
        self.n_positions = 512 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 't5' in self.args.mask_filling_model_name:
            self.mask_model = self.load_mask_model()
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.mask_filling_model_name, model_max_length=self.n_positions, cache_dir=self.args.cache_dir)
        self.query = None
        self.response = None
        self.verbose = True
        self.init_score = None
        self.start_idx = 4
        self.prefix = ""
        
        self.cached_replaced_tokens = set()
        self.original_tokens = set()

    def load_mask_model(self):
        int8_kwargs = {}
        half_kwargs = {}
        if self.args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {self.args.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.args.mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=self.args.cache_dir)
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 512
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()
        if not self.args.int8:
            mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return mask_model
    
    def tokenize_and_mask(self, text, span_len, pct, ceil_pct=False):
        tokens = text.replace('\n', ' \n').split(' ')
        mask_string = '<<<mask>>>'
        # only mask one span
        n_spans = self.args.n_spans

        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start_pos = 0 # only need to prevent moefiying the instruction for chat models as they repeat Q-A.
            # if self.args.dataset == "c4_realnews":
            #     start_pos = 0
            # else:
            # # chat models might repeat the Q:.... A: prompt. So avoid query being perturbed.
            #     start_pos = len(self.prefix.replace('\n', ' \n').split(' ')) 
            start = np.random.randint(start_pos, len(tokens) - span_len)

            end = start + span_len
            search_start = max(0, start - self.args.buffer_size)
            search_end = min(len(tokens), end + self.args.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                # record/remove already masked tokens
                masked_tokens = set(tokens[start:end])
                if len(masked_tokens) > 1:
                    self.cached_replaced_tokens |= masked_tokens
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        n_expected = count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        min_len = int(np.ceil(self.args.span_len * self.args.n_spans * 1.5)) 
        max_len = int(self.args.span_len*self.args.n_spans*2)
        print("min length: ", min_len, "max length: ", max_len)
        outputs = self.mask_model.generate(**tokens, max_length=max_len, min_length=min_len, do_sample=True, top_p=self.args.mask_top_p, num_return_sequences=1, repetition_penalty=self.args.repetition_penalty, eos_token_id=stop_id)  # 500 max, 150
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def perturb_texts_(self, texts, span_len, pct, ceil_pct=False):
        masked_texts = []
        for x in texts:
            masked_texts.append(self.tokenize_and_mask(x, span_len, pct, ceil_pct))

        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_len, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        
        return perturbed_texts

    def perturb_texts_t5(self, texts, span_len, pct, k=5, ceil_pct=False):
      chunk_size = self.args.chunk_size
      if '11b' in self.args.mask_filling_model_name:
          chunk_size //= 2

      outputs = []
      # set chunk_size as 1 to help make sure each original token is replaced.
      for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
          outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_len, pct, ceil_pct=ceil_pct))
      return outputs

    def paraphrase(self, texts, k=5):
        return self.perturb_texts_t5(texts, span_len=self.args.span_len, pct=0.2, k=k, ceil_pct=False)

class Trainer():
    def __init__(self, data, oracle, verbose=True):
        self.data = data
        self.oracle = oracle
        self.verbose = verbose
        self.n_resample = 100
        self.steps = 10
        self.args = get_cmd_args()
        self.mask_filling_model_name = self.args.mask_filling_model_name
        self.n_positions = 512 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.responses = []

    def load_mask_model(self):
        # mask filling t5 model
        int8_kwargs = {}
        half_kwargs = {}
        if self.args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {self.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=self.args.cache_dir)
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 512
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()
        mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return mask_model
    
    def random_walk_attack(self, oracle, attacker, trial_id):

        response = oracle.response
        dist = -1
        n_iter, max_rnd_steps = 0, 200
        # Keeps track of the number of successful perturbations.
        rnd_walk_step = 0

        # Original tokens keeps track of the content of the original response.
        # This is used to determine how much progress we're making when perturbing.
        attacker.original_tokens = set(response.replace("\n", " ").split(" ")) 


        threshold_dist = int(self.args.dist_alpha * len(attacker.original_tokens))
        # This determines how often we're going to checkpoint.
        checkpoint_dist = int(self.args.checkpoint_alpha * len(attacker.original_tokens))
        maintain_quality_or_not = True
        patience = 0
        ckpt_cnt = 0
        mixing_patience = 0
        intermediate_examples = [response]

        all_perturbations = []
        while n_iter < self.args.step_T:
            last_replaced_tokens = set()
            attacker.cached_replaced_tokens = set()
            score = 0
            n_response = attacker.paraphrase([response], self.args.span_len)[0]
            n_response = re.sub(r'\s{2,}', ' ', n_response) # strip the extra spaces
            if oracle.maintain_quality(n_response, model=self.args.oracle_model, tie_threshold=self.args.tie_threshold):
                response = n_response
                # Increment the number of quality-preserving perturbations.
                rnd_walk_step += 1

                # Remove tokens that have been replaced from original tokens
                attacker.original_tokens -= attacker.cached_replaced_tokens

                last_replaced_tokens = attacker.cached_replaced_tokens

                # Reset patience since we just had a successful attempt.
                patience = 0

                if int(rnd_walk_step*self.args.span_len) // checkpoint_dist > ckpt_cnt:
                    intermediate_examples.append(n_response)
                    ckpt_cnt += 1
                print("Get a better response.")
            n_iter += 1
            # dist = Levenshtein.distance(response, oracle.response)
            if n_iter % 10 == 0:
                print("Original Text: ")
                print(oracle.response.__repr__())
            print(f"Walk {rnd_walk_step} / Iteration {n_iter}, {len(attacker.original_tokens)} > {threshold_dist} unique tokens replaced, Paraphrased Text:")
            print(n_response.__repr__())
            self.responses.append(n_response.__repr__())

            perturbation_data = {"trial_id" : trial_id, "step_num": n_iter, "perturbed_text": n_response, "quality_score" : oracle.latest_mean_score}
            all_perturbations.append(perturbation_data)
            
            save_interval = 1 # can choose to save after every few iterations
            if n_iter % save_interval == 0:
                save_intermediate_results(all_perturbations, intermediate_file)
                all_perturbations = []  # Clear the list after saving

            if patience >= 150:
                print("We had 150 unsuccessful attempts at perturbing. Exiting.")
                maintain_quality_or_not = False
                break
            if rnd_walk_step >= max_rnd_steps:
                print("Max random walk steps reached. Exiting.")
                break 

            if len(attacker.original_tokens) <= threshold_dist:
                mixing_patience += 1
            if mixing_patience > self.args.step_T/3:
                print("Mixing patience exceeded. Exiting.")
                break
            
            # If we haven't had any success at perturbing after trying 30 times, backtrack.
            if patience > 30:
                print("Remaining tokens to be masked.")
                print(attacker.original_tokens)
                print("Patience exceeded. Backtrack.")
                response = intermediate_examples[-1]
                attacker.original_tokens = last_replaced_tokens | attacker.original_tokens

            patience += 1

        # Make sure we save the last batch of perturbations.
        save_intermediate_results(all_perturbations, intermediate_file)

        if self.verbose:
            print("Step: ", n_iter)
            print("Original Text: ")
            print(oracle.response.__repr__())
            print("Paraphrase: ")
            print(response.__repr__())
            print("Quality: ")
            print(score)
            print(f"Quality maintained: {maintain_quality_or_not}")
            print()
            
        result_dict = {"watermarked_response": oracle.response, "paraphrased_response": response, "maintain_quality_or_not": maintain_quality_or_not, "patience": patience}
        if len(intermediate_examples) > 1: # intermediate steps for checkpointing etc
            result_dict["intermediate_examples"] = intermediate_examples

        return result_dict

def run_once(query, response=None):
    args = get_cmd_args()
    args.dataset == 'lfqa'
    attacker = Attacker()
    result_dict = {}
    result_dict["watermarked_response"] = response

    attack_results = []
    oracle = Oracle(query, response, check_quality=args.check_quality, choice_granularity=args.choice_granularity, cache_dir=args.cache_dir)
    print(f"Query: {query}")
    data = None
    trainer = Trainer(data, oracle, args)
    result_dict = trainer.random_walk_attack(oracle, attacker)
    paraphrased_response = result_dict["paraphrased_response"]
    print(f"Response: {response}")
    print(f"Paraphrased Response: {paraphrased_response}")
    # result_dict["answer"] = answer
    result_dict["query"] = query
    attack_results.append(result_dict)
    print("Final results:")
    print(attack_results)

def main(query, response=None, trial_id = None):
    args = get_cmd_args()
    args.dataset = 'c4_realnews'
    attacker = Attacker()
    watermark_scheme = args.watermark_scheme
    dataset = args.dataset 
    gen_len = args.gen_len
    out_folder = 'outs_300'
    data = [
        {
            "query": query,
            "output_with_watermark": response
        }
    ]
    # load more from the jsonl file...
    # prefix = f'{watermark_scheme}-watermark/{out_folder}' 
    # data = load_data(f"{prefix}/{dataset}_{watermark_scheme}.jsonl") 
    attack_results = []
    print(args)
    for i, datum in tqdm(enumerate(data), desc="Data Iteration"):
        response = datum["output_with_watermark"]

        if "prefix" in list(datum.keys()):
            query = datum["prefix"]
        elif "query" in list(datum.keys()):
            query = datum["query"]
        else:
            query = None
        
        attacker.prefix = query
        oracle = Oracle(query, response, check_quality=args.check_quality, choice_granularity=args.choice_granularity, use_chat_arena_prompt=True)
        print(f"Iteration {i}-th data:")
        print(f"Query: {query}")
        trainer = Trainer(data, oracle, args)
        result_dict = trainer.random_walk_attack(oracle, attacker, trial_id)
        paraphrased_response = result_dict["paraphrased_response"]
        print(f"Response: {response}")
        print(f"Paraphrased Response: {paraphrased_response}")
        result_dict["watermarked_response"] = datum["output_with_watermark"]
        result_dict["query"] = query

        append_dict_to_json_file('./result_dicts.json', result_dict)

        attack_results.append(result_dict)
        
        # Saved intermediate results
        '''output_file = f"{out_folder}/{dataset}/{dataset}_{watermark_scheme}_len{gen_len}_step{args.step_T}_attack.jsonl"
        print(f"Saving to {output_file}")
        with jsonlines.open(output_file, mode='w') as writer:
            for item in attack_results:
                writer.write(item)'''
    print("Final results:")
    print(attack_results)
    return trainer.responses

if __name__ == '__main__':

    query = "Write me a good story."

    # First story.

    response_1 ="""
    Once upon a time in a mystical forest, there lived a young girl named Elara, who had the unique ability to communicate with animals. Elara's best friend was a wise old owl named Hoot, who had seen many seasons pass in the forest.
    One day, the tranquility of the forest was disturbed by a strange rumbling sound. Elara and Hoot discovered that a giant machine, driven by people from the city, was cutting down the trees. The forest creatures were in panic, and their home was in danger.
    Determined to save the forest, Elara decided to seek the help of the legendary Green Dragon, known to be the guardian of nature. Despite being warned of the dragon's fierce nature, Elara and Hoot ventured deep into the unexplored parts of the forest.
    After days of journeying, they finally found the Green Dragon in a hidden valley. The dragon was initially distrustful, but Elara's genuine concern for the forest and her ability to speak with animals convinced the dragon of her sincerity.
    The Green Dragon agreed to help and revealed an ancient secret to Elara – a magical song that could awaken the spirits of the forest. Elara, with the help of Hoot and the forest animals, sang the magical song under the full moon.
    Miraculously, the spirits of the forest awoke. The trees began to move, gently at first, then with purpose. They formed a barrier, halting the progress of the machines. The people from the city, witnessing this extraordinary event, realized the importance of the forest and the error of their ways.
    From that day on, the forest was protected, and the animals lived in peace. Elara became known as the Guardian of the Forest, and the Green Dragon, once feared, was celebrated as its protector. Elara and Hoot continued to watch over the forest, ensuring its safety and harmony for many years to come.
    And so, the forest remained a magical place, where the spirits danced in the moonlight, and the voice of a young girl who spoke for the trees echoed in the wind, reminding all of the delicate balance between humans and nature.
    """

    # Second story.

    response_2="""
    One stormy night, as thunder roared and waves crashed against the cliffs, Elias noticed a strange glimmer in the water. Braving the storm, he descended from the lighthouse to investigate. There, amidst the tumultuous waves, he found a glowing, ancient bottle sealed with a wax emblem unknown to him. Inside the bottle was a tattered map, leading to a hidden cove on the far side of the island.
    Driven by curiosity and a sense of adventure that he hadn’t felt in years, Elias embarked on a journey to uncover the secrets of the map. He traversed dense forests, scaled steep cliffs, and navigated through hidden trails. Along the way, he encountered a variety of creatures – some friendly, like the wise old owl who offered guidance, and others not so much, like the sly fox that tried to lead him astray.
    After several days of travel, Elias arrived at the hidden cove. The cove was breathtaking, with crystal-clear waters and a beach of fine, white sand. At the center of the cove, half-buried in the sand, was an ancient chest. With trembling hands, Elias opened it to reveal its contents: a collection of rare, luminescent pearls and a note. The note was from a pirate captain who, centuries ago, had hidden his treasure in the cove, regretting his life of plunder and hoping someone worthy would find it.
    Elias returned to the village, his life forever changed by the adventure. He used the pearls to better the lives of the villagers, funding schools, repairing homes, and ensuring the village prospered. The old lighthouse keeper, who had once watched over the sea, became a guardian of the village, his story inspiring generations to come.
    As for the lighthouse, it continued to shine brightly, a symbol of hope and guidance, much like Elias himself, whose journey had shown that it’s never too late for adventure and that the greatest treasures in life are often found in the journey, not the destination.
    """


    # Lord of the Rings.

    lor_prompt = "Write a 500 word essay on the role of power and its impact on characters in the Lord of the Rings series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?"

    lor_1 = """Title: The Role of Power and Its Impact on Characters in the Lord of the Rings Series
    Introduction:
    J.R.R. Tolkien's epic fantasy series, "The Lord of the Rings," delves deep into the complex interplay of power and its consequences on the characters inhabiting Middle-earth. At the heart of this narrative is the One Ring, a symbol of absolute power, and the various characters who come into contact with it. Through their journeys and struggles, Tolkien offers profound insights into the nature of power, its corrupting influence, and the transformative effects it has on individuals and societies.
    The One Ring as a Symbol of Power:
    The One Ring is the central symbol of power in Tolkien's universe. Forged by the Dark Lord Sauron, it possesses the ability to dominate the minds and wills of those who possess it. The Ring represents the corrupting allure of power, as it promises its bearer immense control and authority. It becomes the object of desire for many characters throughout the series, illustrating the seductive nature of power.
    Corruption of Characters:
    One of the most prominent examples of the corrupting influence of power is seen in the character of Gollum. Originally, Gollum was a hobbit named Sméagol, who came into possession of the Ring. Over time, the Ring twisted his nature, turning him into a creature consumed by his obsession with the Ring. Gollum's transformation is a poignant representation of how power can corrupt even the most innocent of souls.
    Another character profoundly affected by the lure of power is Boromir, a noble warrior from Gondor. Boromir initially joins the Fellowship with noble intentions, but his desire to use the Ring for the defense of his people ultimately leads to his downfall. His inner struggle and eventual redemption highlight the dangers of succumbing to the temptation of power.
    The Ring-Bearer, Frodo Baggins, experiences the corrupting influence of the Ring firsthand. Despite his pure and noble heart, he is not immune to its allure. Throughout his journey, Frodo faces inner turmoil and temptation, which serves as a testament to the insidious nature of power.
    The Nature of Power
    Tolkien's portrayal of power in "The Lord of the Rings" suggests several key insights into its nature. First, power is a double-edged sword. While it offers the potential for great good, it also carries the risk of immense corruption. The Ring's ability to corrupt its bearers underscores the idea that absolute power corrupts absolutely.
    Second, power is a test of character. Characters like Frodo and Sam demonstrate that even in the face of overwhelming power, individuals can choose to resist its corrupting influence and act with courage and selflessness. Their unwavering commitment to destroying the Ring reflects Tolkien's belief in the capacity of individuals to resist the temptations of power.
    Additionally, Tolkien emphasizes the importance of collective action in the face of power. The Fellowship of the Ring, comprised of diverse races and backgrounds, exemplifies the idea that unity and cooperation are essential when confronting the most formidable manifestations of power. Together, they stand a better chance of resisting the Ring's influence.
    Conclusion:
    In "The Lord of the Rings," J.R.R. Tolkien masterfully explores the role of power and its profound impact on characters and societies. Through the symbolism of the One Ring and the character arcs of individuals like Gollum, Boromir, and Frodo, Tolkien reveals the corrupting influence of absolute power and the enduring struggle between the desire for power and the need for moral integrity.
    Tolkien's work suggests that power is a force that can shape destinies, test character, and ultimately determine the course of history. However, he also offers hope through the examples of characters who resist the seductive pull of power, illustrating that the choices individuals make can lead to redemption and the preservation of the values that define a just and noble society. Ultimately, "The Lord of the Rings" is a timeless exploration of the timeless question: What does power do to those who wield it, and what does it take to resist its temptations?"""

    lor_2 = """Title: The Corrupting Influence of Power in "The Lord of the Rings"
    Introduction
    J.R.R. Tolkien's epic fantasy series, "The Lord of the Rings," explores the profound and often destructive influence of power on its characters. At the heart of this narrative is the One Ring, a symbol of ultimate power that serves as a catalyst for corruption and moral dilemmas. Tolkien's work delves deeply into the nature of power, offering insights into how it can shape individuals and societies and the consequences of wielding it.
    The Symbolism of the Ring
    The One Ring in "The Lord of the Rings" is the quintessential symbol of power. Crafted by the Dark Lord Sauron, the Ring contains a portion of his own power, which he uses to dominate Middle-earth. The Ring's allure is undeniable, as it grants immense power to its possessor. However, it comes at a great cost. The Ring is not just a magical artifact; it embodies the corrupting nature of power itself. As characters are drawn to its power, they begin to experience its malevolent influence, demonstrating the allure and peril of power.
    The Impact on Characters
    Throughout the series, various characters grapple with the temptation of the Ring and its corrupting effects. The most prominent example is Frodo Baggins, the Ring-bearer. Initially, Frodo resists the Ring's pull, but as he carries it closer to its place of destruction, the burden takes a toll on him. The Ring seeks to dominate its possessor, making them more susceptible to its influence. Frodo's struggle embodies the idea that even the most virtuous individuals can be corrupted by power.
    Another character profoundly affected by the allure of power is Boromir. His desire to use the Ring to defend his homeland ultimately leads him to attempt to take it from Frodo. Boromir's tragic fall demonstrates how power can corrupt even those with noble intentions. It is a cautionary tale of how power can twist one's values and goals.
    Gollum, the Ring's previous owner, serves as a tragic example of the Ring's corrupting influence. Once a normal hobbit-like creature named Sméagol, he was consumed by his obsession with the Ring over centuries, becoming a grotesque and tormented figure. Gollum's transformation underscores the corrosive nature of power, as it destroyed his humanity and left him a wretched creature driven solely by his lust for the Ring.
    Societal Impact
    Beyond individual characters, "The Lord of the Rings" also explores the impact of power on society. The quest for power is a recurring theme, with various factions seeking control of the Ring for their own purposes. The Dark Lord Sauron seeks to dominate Middle-earth using the Ring's power, and this desire for supremacy leads to war and suffering on a grand scale. The struggles for power among races and nations highlight the destructive potential of unchecked ambition.
    Tolkien's Insights on Power
    Through the symbolism of the Ring and the experiences of its characters, Tolkien offers profound insights into the nature of power. He suggests that power is inherently corrupting, and its allure can lead individuals astray. Furthermore, Tolkien emphasizes the importance of humility and selflessness in the face of power, as these qualities are crucial in resisting its corrupting influence.
    Conclusion
    In "The Lord of the Rings," J.R.R. Tolkien masterfully explores the role of power and its impact on characters and societies. The One Ring serves as a potent symbol of power's allure and corruption, leading individuals like Frodo, Boromir, and Gollum down dark paths. Tolkien's work is a cautionary tale that reminds us of the dangers of unchecked ambition and the importance of resisting the corrupting influence of power. Ultimately, "The Lord of the Rings" offers profound insights into the human condition and the enduring struggle between the desire for power and the need for humility and selflessness."""


    args = get_cmd_args()

    intermediate_file = args.intermediate
    output_file = args.output
    input_file = args.input

    if input_file is not None:
        print(f"Successfully read the input file {input_file}.")
        df_in = pd.read_csv(input_file)
        query = df_in['query'][0]
        responses = list(df_in['response'])
    else:
        responses = [response_1, response_2]

    query = lor_prompt
    responses = [lor_1, lor_2]

    data = []

    num_trials = args.num_trials

    for trial_id in range(1, 1 + num_trials):
        # Loop over every query and response and get the perturbed responses
        perturbed_responses = [main(query=query, response=response, trial_id = trial_id) for response in responses]

        # Put the perturbed responses in the DF using the schema
        for i, random_walk in enumerate(perturbed_responses, 1):
            for step_num, response in enumerate(random_walk, 1):
                data.append((trial_id, i, step_num, response))
    
    # Create the Pandas DF and write it to a CSV file
    df_out = pd.DataFrame(data, columns=['trial_id', 'story_id','step_num', 'response'])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)