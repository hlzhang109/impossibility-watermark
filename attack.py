
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
    
    def random_walk_attack(self, oracle, attacker):
        # find a perturbation (through repeated sampling) such that the quality oracle says quality is maintained; then repeat, several times 
        response = oracle.response
        dist = -1
        n_iter, max_rnd_steps = 0, 200
        rnd_walk_step = 0
        # set the stopping criterion
        threshold_dist = self.args.dist_alpha * len(oracle.response)
        attacker.original_tokens = set(response.replace("\n", " ").split(" ")) 
        threshold_dist = int(self.args.dist_alpha * len(attacker.original_tokens))
        checkpoint_dist = int(self.args.checkpoint_alpha * len(attacker.original_tokens))
        maintain_quality_or_not = True
        patience = 0
        ckpt_cnt = 0
        mixing_patience = 0
        intermediate_examples = [response]
        while n_iter < self.args.step_T:
            last_replaced_tokens = set()
            attacker.cached_replaced_tokens = set()
            score = 0
            n_response = attacker.paraphrase([response], self.args.span_len)[0]
            n_response = re.sub(r'\s{2,}', ' ', n_response) # strip the extra spaces
            if oracle.maintain_quality(n_response, model=self.args.oracle_model, tie_threshold=self.args.tie_threshold):
                response = n_response
                rnd_walk_step += 1
                attacker.original_tokens -= attacker.cached_replaced_tokens
                last_replaced_tokens = attacker.cached_replaced_tokens
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
            if rnd_walk_step >= max_rnd_steps or patience >= 150:
                print("Max random walk steps reached. Exiting.")
                break 
            if len(attacker.original_tokens) <= threshold_dist:
                mixing_patience += 1
            if mixing_patience > self.args.step_T/3:
                print("Mixing patience exceeded. Exiting.")
                break
            if patience > 30:
                print("Remaining tokens to be masked.")
                print(attacker.original_tokens)
                print("Patience exceeded. Backtrack.")
                response = intermediate_examples[-1]
                attacker.original_tokens = last_replaced_tokens | attacker.original_tokens
            patience += 1

        if patience >= 150:
            maintain_quality_or_not = False
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
    oracle = Oracle(query, response, check_quality=args.check_quality, choice_granuality=args.choice_granularity, cache_dir=args.cache_dir)
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

def main(query, response=None):
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
        oracle = Oracle(query, response, check_quality=args.check_quality, choice_granuality=args.choice_granularity)
        print(f"Iteration {i}-th data:")
        print(f"Query: {query}")
        trainer = Trainer(data, oracle, args)
        result_dict = trainer.random_walk_attack(oracle, attacker)
        paraphrased_response = result_dict["paraphrased_response"]
        print(f"Response: {response}")
        print(f"Paraphrased Response: {paraphrased_response}")
        result_dict["watermarked_response"] = datum["output_with_watermark"]
        result_dict["query"] = query
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

    output_file = get_cmd_args().output
    input_file = get_cmd_args().input

    if input_file is not None:
        df_in = pd.read_csv(input_file)
        query = df_in['query'][0]
        responses = list(df_in['response'])
    else:
        responses = [response_1, response_2]
    
    perturbed_responses = [main(query=query, response=response) for response in responses]
    perturbed_walks = { f'response_{i}' : random_walk for i, random_walk in enumerate(perturbed_responses, 1)}
    
    df_out = pd.DataFrame(perturbed_walks)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)