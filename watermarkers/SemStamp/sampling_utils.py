import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import StoppingCriteria
from nltk.tokenize import sent_tokenize
from string import punctuation
from itertools import groupby
import re
from typing import *

MAX_TRIALS = 100
if torch.cuda.is_available():
    rng = torch.Generator("cuda")
else: 
    rng = torch.Generator("cpu")
hash_key = 15485863
PUNCTS = '!.?'
device = "cuda" if torch.cuda.is_available() else "cpu"

def handle_bullet_points(sentences):
    new_sentences = []
    digit_pattern = re.compile(r'^\d+\.$')
    
    i = 0
    while i < len(sentences) - 1:
        if digit_pattern.match(sentences[i].strip()):
            new_sentences.append(f'{sentences[i].strip()} {sentences[i + 1]}')
            i += 1  # Skip the next element as it's already added
        else:
            new_sentences.append(sentences[i])
        i += 1

    # Add the last sentence as well
    new_sentences.append(sentences[-1])
    
    return new_sentences

def tokenize_sentences(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    processed_sentences = handle_bullet_points(sentences)
    return processed_sentences

# TODO: Remove this if the other solution works.
def is_candidate_text_one_sentence(self, candidate_text: str) -> bool:
    sentences = sent_tokenize(candidate_text)
    num_sentences = len(sentences)

    if num_sentences == 2:
        return re.match(r'^\d\.$', sentences[0].strip())

    return num_sentences == 1

class SentenceEndCriteria(StoppingCriteria):
    """
    ONLY WORK WITH BATCH SIZE 1

    Stop generation whenever the generated string is **more than one** sentence (i.e. one full sentence + one extra token).
    This is determined using a slight modification of sent_tokenize.
    Only stop if ALL sentences in the batch is at least two sentences

    Args:
        tokenizer (PreTrainedTokenizer):
            The exact tokenizer used for generation. MUST BE THE SAME!
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0

    def update(self, current_text):
        self.current_num_sentences = len(tokenize_sentences(current_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Ensure that the batch size is 1.
        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        sentences = tokenize_sentences(text)
        num_sentences = len(sentences)

        return num_sentences > self.current_num_sentences + 1

def discard_final_token_in_outputs(outputs):
    # Discard the final token in the sequences within the 'outputs' object.
    # Assuming 'outputs.sequences' is a 2D array where each row is a sequence and each column is a token,
    # this line removes the last token from each sequence.
    outputs.sequences = outputs.sequences[:, :-1]  # (bz, seqlen)
    return outputs

def extract_prompt_from_text(text, len_prompt):
    tokens = text.split(' ')
    tokens = tokens[:len_prompt]
    new_text = ' '.join(tokens)
    prompts = []
    for p in PUNCTS:
        idx = new_text.find(p)
        if idx != -1:
            tokens = new_text[:idx + 1].split(" ")
            # has to be greater than a minimum prompt
            if len(tokens) > 3:
                prompts.append(new_text[:idx + 1])
    if len(prompts) == 0:
        prompts.append(new_text + ".")
    # select first (sub)sentence, deliminated by any of PUNCTS
    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
    return prompt

def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    # Generate text using the model with the given configuration and stopping criteria.
    outputs = model.generate(
            text_ids,
            gen_config,
            stopping_criteria=stopping_criteria,
        )
    
    outputs = discard_final_token_in_outputs(outputs)
    new_text_ids = outputs.sequences
    new_text = tokenizer.decode(
        new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    return new_text, new_text_ids

def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]
