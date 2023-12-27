
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
    def __init__(self, data, oracle, intermediate_file, verbose=True):
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
        intermediate_file = self.args.intermediate
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
    attacker = Attacker()

    watermark_scheme = args.watermark_scheme
    dataset = args.dataset 
    gen_len = args.gen_len

    output_file = args.output
    input_file = args.input
    stats_file = args.result_stats

    num_trials = args.num_trials

    use_query = args.use_query

    # Try to read the input file
    if input_file is not None:
        print(f"Successfully read the input file {input_file}.")
        df_in = pd.read_csv(input_file)
        queries = list(df_in['query'])
        responses = list(df_in['response'])
    else:
        print(f"Couldn't find input file {input_file}")
        return 1

    input_data = [{'query': q, 'response': r} for q, r in zip(queries, responses)]

    output_data = []

    attack_results = []
    print(args)
    for trial_id in range(1, num_trials+1):
        for i, datum in tqdm(enumerate(input_data), desc="Data Iteration"):
            response = datum["output_with_watermark"]

            if "prefix" in list(datum.keys()):
                query = datum["prefix"]
            elif "query" in list(datum.keys()):
                query = datum["query"]
            else:
                query = None
            
            attacker.prefix = query
            oracle = Oracle(query, response, use_query=use_query, check_quality=args.check_quality, choice_granularity=args.choice_granularity, use_chat_arena_prompt=True)
            print(f"Iteration {i}-th data:")
            print(f"Query: {query}")
            trainer = Trainer(input_data, oracle, args)
            result_dict = trainer.random_walk_attack(oracle, attacker, trial_id)
            paraphrased_response = result_dict["paraphrased_response"]
            print(f"Response: {response}")
            print(f"Paraphrased Response: {paraphrased_response}")
            result_dict["watermarked_response"] = datum["output_with_watermark"]
            result_dict["query"] = query

            # Add the stats of the last attack to the JSON file
            append_dict_to_json_file(stats_file, result_dict)

            attack_results.append(result_dict)

            # Put the perturbed responses in the DF using the schema
            for i, random_walk in enumerate(trainer.responses, 1):
                for step_num, response in enumerate(random_walk, 1):
                    output_data.append((trial_id, i, step_num, response))
            
    print("Final results:")
    print(attack_results)

    # Create the Pandas DF and write it to a CSV file
    df_out = pd.DataFrame(output_data, columns=['trial_id', 'story_id','step_num', 'response'])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)

    return 0