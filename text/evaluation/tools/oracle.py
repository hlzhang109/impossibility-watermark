import json
import openai
import os
import numpy as np
import tiktoken
import random
from time import sleep
import jsonlines
import re
import copy
import transformers
import torch
from transformers import AutoTokenizer, pipeline

DEF_MODEL = "gpt-4"
MODELS = {"gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TOKENIZERS  = {model : tiktoken.encoding_for_model(MODELS[model]) for model in MODELS }
openai.api_key = os.environ["OPENAI_API_KEY"]

def set_seed(seed):
   random.seed(seed)
   np.random.seed(seed)

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    distance = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            distance += 1      
    return distance

def tokens(s, model = DEF_MODEL):
  """Returns tokens of a string.
     Returns two lists, one of strings and the other of integers."""
  tokenizer = TOKENIZERS[model]
  L=tokenizer.encode(s)
  return [str(tokenizer.decode_single_token_bytes(t))[2:-1] for t in L],L

def count_tokens(s, tokenizer, model = DEF_MODEL):
  """Count the number of tokens in a string"""
  return len(tokenizer.encode(s))

def truncate(s,n, model = DEF_MODEL):
  """Truncase to n tokens"""
  tokenizer = TOKENIZERS[model]
  L = tokenizer.encode(s)
  return tokenizer.decode(L[:n])

def tokens2str(tokens, model = DEF_MODEL):
  tokenizer = TOKENIZERS[model]
  """Returns string from tokens (should get the integer tokens)"""
  return tokenizer.decode(tokens)

def read_jsonl(file: str):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def query_openai(prompt, model="text-davinci-003", max_tokens=512):
    # prompt = instruction+"\n"+query
    response = openai.Completion.create(
        engine=model, # "gpt-3.5-turbo-instruct"
        prompt=prompt,
        temperature=.2,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        # stop=["\n"]
    )
    return response.choices[0].text

def chopped(s,k=30):
  """Chop a string to a shorter representation for prining"""
  if len(s)<=2*k: return(s)
  return f"{s[:k]}...{s[-k:]}"

def chat(message, history = [{"role": "system", "content": "You are a research assistant."}],
         model = "gpt-4", # model is "gpt-3" or "gpt-4"
         return_more = False,  # return also history and response object
         debug=True,  # print message and response
         supress_exception = False,  # supress exception and return None
         retries = 500, # number of times to retry if there is an exception
         tokenizer = None,
         **extra_params # extra parameters for Chat
         ):
  """General routine to send a message to GPT.
     Can take an optional parameter history of messages, and can also return message and history as extra parameter"""
  CONTEXT = {"gpt-4":8192, "gpt-3.5": 4096}
  if tokenizer is None:
    tokenizer = TOKENIZERS[model]
  hist_tokens  = count_tokens(", ".join(D["content"] for D in history), tokenizer)
  message_tokens = count_tokens(message, tokenizer)

  while retries >= 0:
    try:
      if debug: print(f"Message:\n {chopped(message)} ({message_tokens} tokens, history {hist_tokens} tokens)", flush=True)
      history = history + [{"role": "user", "content": f"{message}"}]
      params = dict(
            model = MODELS[model],
            messages= history,
            max_tokens=1024, # 512
            n=1,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            )
      params.update(extra_params) # update based on any extra parameters to add
      response = openai.ChatCompletion.create(**params)
      break
    except Exception as e:
      print(f"Error!:\n{e}")
      if retries:
        print(f"Retrying: {retries} tries left")
        sleep(1)
        retries -= 1
      elif not supress_exception:
        raise e
      else:
        return None

  text_response =  response.choices[0]['message']['content']
  if debug: print(f"Response:\n {chopped(text_response)} ({count_tokens(text_response, tokenizer)} tokens)", flush=True)
  if return_more:
    return text_response, history + [{"role": "assistant", "content": text_response}], response
  return text_response
    
class QualityOracle:
    def __init__(self, model_name_or_path, check_quality="checker", choice_granularity=5, use_chat_arena_prompt=False, device="cuda") -> None:
        self.init_score = -1
        self.detailed_prompt = "" 
        self.check_quality = check_quality
        self.choice_granularity = choice_granularity
        self.system_prompt = "You are a capable, helpful and useful assistant." if not use_chat_arena_prompt else self.chat_arena_prompt
        self.history =  [{"role": "system", "content": self.system_prompt}]
        self.device = device
        if model_name_or_path == "OpenAssistant/reward-model-deberta-v3-large-v2":
            self.rm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
            self.reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
        elif model_name_or_path == "sfairXC/FsfairX-LLaMA3-RM-v0.1":
            self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.rm_pipe = pipeline(
                "sentiment-analysis",
                model=model_name_or_path,
                device=device,
                tokenizer=self.rm_tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16}
            )
            self.pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 1
            }

    @property
    def chat_arena_prompt(self):
        return f'''
        Please act as an impartial judge and evaluate the quality of the responses provided by two
        AI assistants to the user question displayed below. You should choose the response that
        follows the user’s instructions and answers the user’s question better. Your evaluation
        should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
        and level of detail of their responses. Begin your evaluation by comparing the two
        responses and provide an explanation. Avoid any position biases and ensure that the
        order in which the responses were presented does not influence your decision. Do not allow
        the length of the responses to influence your evaluation. Be as objective as possible.
        Make sure the better resopnse does not have isolated punctuation marks.
        Note that grammatical errors would greatly degrade the quality of a response.
        '''
    
    def instruction(self, query):
        return f"Below are two candidate responses to the query {query}: "

    @property
    def comparison_prompt(self):
        return '''Response A: [A] \n Response B: [B] \n 
        Compare which of the two above responses is a better response to the given prompt. 
        Explain your reasoning step by step. 
        Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses.
        Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
        '''   
    
    @property
    def check_error_prompt(self):
        return '''Does the response above have any grammatical errors or capitalization errors or punctuation mistakes? If so, answer 1, otherwise answer 2. '''
    
    @property
    def check_quality_prompt(self):
        return  "Text quality is affected by factors such as unnecessary repetitions, grammar, coherence, and accuracy of the responses. Especially, having grammatical errors, repetitiveness, capitalization errors or punctuation mistakes would greatly degrade the quality of a response." + "\n" + \
                "Therefore, is the new modified response of equal or higher quality compared to original response? If so, answer Yes, otherwise answer No."
    
    @property
    def five_choice(self):
        return '''
        (1) Response A is much better than response B
    	(2) Response A is a little better than response B
    	(3) Responses A and B have similar quality
    	(4) Response B is a little better than response A
    	(5) Response B is much better than response A
        '''

    @property
    def three_choice(self):
        return '''
        (1) Response A is better than response B
    	(2) Responses A and B have similar quality
    	(3) Response B is better than response A
        '''
    
    @property
    def answer_prompt(self):
        choices = self.five_choice if self.choice_granularity == 5 else self.three_choice
        return f'''So based on your reasoning, choose one of the following {choices}''' # So which of the two responses is a better response to the given prompt? 
    
    def query_gpt_once(self, prompt, model="gpt-3.5", tokenizer=None):
        # avoid using max_tokens which can be too short for generating explanations.
        n_attempt = 0
        while n_attempt < 5:
            try:
                oracle_reasoning = chat(prompt, history=self.history, model=model, tokenizer=tokenizer)
                history = copy.deepcopy(self.history)
                history.append({"role": "user", "content": f"{prompt}"})
                history.append({"role": "assistant", "content": f"{oracle_reasoning}"})
                oracle_answer = chat(self.answer_prompt, history=history, model=model, tokenizer=tokenizer)
                pattern = r'\((\d+)\)'
                filtered_response = re.findall(pattern, oracle_answer)
                print(oracle_reasoning)
                print(oracle_answer)
                score = int(filtered_response[-1])
                if score not in range(1, self.choice_granularity+1):
                    print(f"return choice {score} not in [1, {self.choice_granularity}]")
                    continue
                return score
            except Exception as e:
                print(e)
                n_attempt += 1
                continue

    def query_rm_once_deberta(self, query, response, device="cuda:0"):
        context = '###Human: ' + query + '###Assistant: '
        text = context + response
        tokenized_text = self.rm_tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_mask = tokenized_text['input_ids'].to(device), tokenized_text['attention_mask'].to(device)
        score = self.reward_model(input_ids=input_ids,attention_mask=attention_mask).logits.detach()
        return score

    def query_rm_once_llama(self, query, response, device="cuda:0"):
        chat = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
        test_texts = [self.rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(self.rm_tokenizer.bos_token, "")]
        pipe_outputs = self.rm_pipe(test_texts, **self.pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards[0]
    
    def query_rm_once(self, query, response):
        if self.model_name_or_path == "OpenAssistant/reward-model-deberta-v3-large-v2":
            return self.query_rm_once_deberta(query, response)
        elif self.model_name_or_path == "sfairXC/FsfairX-LLaMA3-RM-v0.1":
            return self.query_rm_once_llama(query, response)
       
    def maintain_quality(self, query, original_response, perturbed_response, tie_threshold=0.02, gpt_model="gpt-3.5"):
        original_score = self.query_rm_once(query, original_response)
        perturbed_score = self.query_rm_once(query, perturbed_response)
        if abs(original_score - perturbed_score) < tie_threshold or perturbed_score >= original_score:
            return True
        else:
            return False

    def check_quality_hf_models(self, prompt):
        inputs = self.rm_tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_outputs = self.rm_model.generate(**model_inputs, max_new_tokens=512, top_p=0.95, num_return_sequences=1)
        output_text = self.rm_tokenizer.batch_decode(model_outputs[:, model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        print(prompt)
        print()
        print(f"quality oracle model assessment: {output_text}")
        print()
        return output_text

    def report_gpt_mean_score(self, original_response, perturbed_response, tie_threshold=0.1, model="gpt-3.5"):
        choice = self.query_gpt_once(perturbed_response, original_response)
        self.choice_granularity = 3
        # NOTE positive scores for paraphrased response winning
        score_dict = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5:-1} if self.choice_granularity == 5 else {1: 1, 2: 0, 3: -1} 
        if choice is None:
            return False
        score = score_dict[choice]
        print()
        print("Second round of comparison:")
        print()
        choice = self.query_gpt_once(original_response, perturbed_response, tie_threshold=tie_threshold)
        second_score = score_dict[choice]
        return (score-second_score)/2 # essentially (1, 0), (1, -1), (1, 1), (0, 1), (0, -1), (0, 0), (-1, 1), (-1, 0), (-1, -1) -> 0.5, 0, 1, 0.5, -0.5, 0, 0, -0.5, -1