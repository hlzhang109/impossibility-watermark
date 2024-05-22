from . import sampling_utils
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, StoppingCriteriaList
from .sbert_lsh_model import SBERTLSHModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from .sampling_utils import SentenceEndCriteria, device, gen_sent
import logging

import textwrap

# rng = torch.Generator()
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)
MAX_TRIALS = sampling_utils.MAX_TRIALS
hash_key = sampling_utils.hash_key

log = logging.getLogger(__name__)

def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))


def get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int):
    n_bins = 2**lsh_dim
    n_accept = int(n_bins * accept_rate)
    rng.manual_seed(hash_key * seed)
    vocab_permutation = torch.randperm(n_bins, device='cuda', generator=rng)
    greenlist_ids = vocab_permutation[:n_accept]
    return greenlist_ids.to(device)


def reject_close_generation(lsh_model, sents, margin, cutoff=None):
    embeds = lsh_model.get_embeddings(sents)
    embeds = torch.tensor(embeds, device='cuda')
    normals = torch.tensor(lsh_model.hasher.normals, device='cuda')
    if cutoff != None:
        normals = normals[:cutoff]

    # sims[i, j] is the cosine similarity between the ith generation and the jth normal vec
    sims = cosine_distance_matrix(embeds, normals)
    sims_abs = torch.abs(sims)
    # max_sim is the highest cosine similarity of each generation with any normal vec
    min_sims = sims_abs.min(dim=1).values
    select = []
    for i in range(len(min_sims)):
        # log.info(max_sims[i])
        min_sim = min_sims[i].item()
        if (abs(min_sim) >= margin):
            # log.info(min_sim)
            select.append(i)
    sents = np.array(sents)
    sents = sents[select]
    return list(sents), select

def mixtral_format_instructions(prompt):
    return textwrap.dedent(f"""
    [INST]
    {prompt}
    [/INST]

    Answer:""")

def lsh_reject_completion(
        prompt: str,
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, gen_config: GenerationConfig,  # gen args
        lsh_model: SBERTLSHModel, lsh_dim: int,  # LSH args
        lmbd=1.0, # watermark args. lambda is probability of accepting (i.e., green list size)
        device='cuda',
        margin=0.002, pipeline=None, generator_kwargs = None, max_new_tokens=None, #TODO: Make this better, just put this here in order to test.
        **kwargs):
    
    if "Mixtral" in model.config._name_or_path:
        prompt = mixtral_format_instructions(prompt)
            
    sent_end_criteria = SentenceEndCriteria(tokenizer)
    sent_end_criteria.update(prompt)

    lsh_seed = lsh_model.get_hash([prompt])[0]
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)

    text = prompt
    new_text = prompt

    text_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    prompt_length = len(text_ids[0])

    total_trials = 0
    success_trials = 0  # also include trials that maxed out MAX_TRIALS
    current_trials = 0
    maxedout_trials = 0
    debug_text_segments = [(prompt, text_ids.size(1), lsh_seed)]
    
    while True:
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        # TODO: This doesn't work right now, since we removed their generation config as it was throwing an error with Mixtral.
        if "opt" in model.config._name_or_path:
            new_text, new_text_ids = gen_sent(model = model, 
                tokenizer = tokenizer, 
                text_ids = text_ids,
                gen_config = gen_config,
                stopping_criteria = stopping_criteria
            )
        # TODO: Experimenting with Mixtral right now, it doesn't work.
        elif "Mixtral" in model.config._name_or_path:
            generator_kwargs['stopping_criteria'] = stopping_criteria
            outputs = model.generate(text_ids, gen_config, **generator_kwargs)
            new_text_ids = outputs.sequences
            new_text = tokenizer.decode(
                new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
        elif "Llama" in model.config._name_or_path:
            #TODO: pipeline outputs is a string, but need to do tokenization on sequences? Not sure how to resolve, will look att later...
            generator_kwargs['stopping_criteria'] = stopping_criteria
            
            # Generate text using the pipeline
            generated_text = pipeline.generate_text(text, generator_kwargs)
            
            # Tokenize the generated text and convert it to a list of IDs
            new_text_ids = tokenizer.encode(generated_text)
            
            # Convert the list of IDs to a tensor to match Mixtral's format
            new_text_ids = torch.tensor([new_text_ids])
            
            # Decode the tokenized text to get the new text
            new_text = tokenizer.decode(new_text_ids[0], skip_special_tokens=True)

            # generator_kwargs['stopping_criteria'] = stopping_criteria
            # outputs = model.generate(text_ids, gen_config, **generator_kwargs)
            # new_text_ids = outputs.sequences
            # new_text = tokenizer.decode(
            #     new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)

        else:
            raise NotImplementedError("model type not supported")
        
        if new_text == '':
            log.info('WARNING: stopped generation because generated nothing (after discarding last generated token)')
            break


        log.info(f"Candidate text: {new_text}")
        log.info(f"Accept Mask: {accept_mask}")

        total_trials += 1
        current_trials += 1

        # Use the LSH model to reject generations too close to the margin.
        accepted_text, _ = reject_close_generation(
                lsh_model, [new_text], margin=margin, cutoff=None)
        
        # If no text is accepted and the current number of trials is less than the maximum allowed trials, continue the loop.
        if (len(accepted_text) == 0 and current_trials < MAX_TRIALS):
            continue
        
        # Get the LSH hash for the generated text, continue if it's not in the correct place
        lsh_candidate = lsh_model.get_hash([new_text])[0]

        log.info(f"LSH Candidate: {lsh_candidate}")

        if lsh_candidate not in accept_mask:
            log.info(f"Candidate text is doesn't fall into the correct place in the embedding space.")
            continue
        else:
            log.info("Candidate text falls within the semantic partition.")
        
        if (lsh_candidate in accept_mask) or current_trials >= MAX_TRIALS:
            if current_trials >= MAX_TRIALS:
                log.info(
                    f'WARNING: desired semantic signature can\'t be sampled after max_trials {MAX_TRIALS}')
                log.info(f'CONTEXT: {text}')
                log.info(
                    f'NOTE: use regular (non-filtered-by-sig) continuation: {new_text}')
                maxedout_trials += 1

            debug_text_segments.append(
                (new_text, new_text_ids.size(1) - text_ids.size(1), lsh_candidate))
            
            success_trials += 1

            # Proceed to the next sentence
            lsh_seed = lsh_candidate
            accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
            text += new_text
            text_ids = new_text_ids
            sent_end_criteria.update(text)
            current_trials = 0

            # If the number of new tokens generated reaches the maximum new token number, stop generating.
            # TODO: Fix the max_new_tokens if it doesn't work.
            if (len(text_ids[0]) - prompt_length) >= max_new_tokens-1:
                break
    return text, total_trials