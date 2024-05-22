import logging
from watermarker import Watermarker

import nltk
import torch
import os
from transformers import LogitsProcessorList, AutoModelForCausalLM, GenerationConfig, AutoTokenizer, AutoConfig, StoppingCriteriaList
from SemStamp.sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from SemStamp.sampling_lsh_utils import get_mask_from_seed, reject_close_generation
from SemStamp.sampling_utils import extract_prompt_from_text, SentenceEndCriteria, gen_sent
from SemStamp.detection_utils import detect_kmeans, detect_lsh
from nltk.tokenize import sent_tokenize

import textwrap



# TODO: This is probably from k-SemStamp. It generates a bug right now.
# from sampling_kmeans_utils import embed_gen_list, get_cluster_centers, kmeans_reject_completion, load_embeds

# TODO: The embed issue after fixing it in their code. What does TODO mean?
# TODO: Can we run with larger models than opt? Is there some part of their code that assumes opt?

nltk.download('punkt')

PUNCTS = '.,!?'

log = logging.getLogger(__name__)

class SemStampWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, is_completion=False):
        super().__init__(cfg, pipeline, n_attempts, is_completion)

    def setup_watermark_components(self):
        # NOTE: currently, no batching.

        is_offline = os.environ.get('TRANSFORMERS_OFFLINE') is not None and os.environ.get(
        'TRANSFORMERS_OFFLINE') == '1'

        # block the LLM from generating
        bad_words_ids = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device='cuda').tolist()

        if 'Llama' not in self.cfg.generator_args.model_name_or_path:
            self.gen_config = GenerationConfig.from_pretrained(
                            self.cfg.generator_args.model_name_or_path,
                            return_dict_in_generate=True,
                            max_new_tokens=self.cfg.generator_args.max_new_tokens,
                            min_new_tokens=self.cfg.generator_args.min_new_tokens,
                            do_sample=self.cfg.generator_args.do_sample,
                            temperature=self.cfg.generator_args.temperature,
                            top_k=self.cfg.generator_args.top_k,
                            bad_words_ids=bad_words_ids,
                            repetition_penalty=self.cfg.generator_args.repetition_penalty,                        
                            # top_p=0.96,
                            local_files_only=is_offline
                    )
        else:
            self.gen_config = None
            self.pipeline._init_pipeline_config(self.cfg.generator_args)

        self.generator_kwargs.update([('bad_words_ids', bad_words_ids), ('min_new_tokens', self.cfg.watermark_args.min_new_tokens)])

        log.info(self.generator_kwargs)
        
        log.info(f"Initializing embedder model.")
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")
        self.embedder.eval()
        log.info(f"Finished initializing embedder model.")

        # TODO: Make the code work with our pipeline setup.
        
        # TODO: Fix lsh_model_path. We're using the default model in their code right now.
        self.lsh_model = SBERTLSHModel(lsh_model_path=None,
                                  device=self.cfg.watermark_args.device, batch_size=1, lsh_dim=self.cfg.watermark_args.sp_dim, sbert_type='base', embedder=self.embedder)
        
        self.model.eval()
        
    def mixtral_format_instructions(self, prompt):
        return textwrap.dedent(f"""
        [INST]
        {prompt}
        [/INST]

        Answer:""")
    
    def generate(self):
        # TODO: This doesn't work right now, since we removed their generation config as it was throwing an error with Mixtral.
        if "opt" in self.model.config._name_or_path:
            new_text, new_text_ids = gen_sent(model = model, 
                tokenizer = self.tokenizer, 
                text_ids = text_ids,
                gen_config = gen_config,
                stopping_criteria = stopping_criteria
            )
        # TODO: Experimenting with Mixtral right now, it doesn't work.
        elif "Mixtral" in model.config._name_or_path:
            generator_kwargs['stopping_criteria'] = stopping_criteria
            outputs = model.generate(text_ids, gen_config, **generator_kwargs)
            new_text_ids = outputs.sequences
            new_text = self.tokenizer.decode(
                new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
        elif "Llama" in model.config._name_or_path:
            #TODO: pipeline outputs is a string, but need to do tokenization on sequences? Not sure how to resolve, will look att later...
            generator_kwargs['stopping_criteria'] = stopping_criteria
            
            # Generate text using the pipeline
            generated_text = pipeline.generate_text(text, generator_kwargs)
            
            # Tokenize the generated text and convert it to a list of IDs
            new_text_ids = self.tokenizer.encode(generated_text)
            
            # Convert the list of IDs to a tensor to match Mixtral's format
            new_text_ids = torch.tensor([new_text_ids])
            
            # Decode the tokenized text to get the new text
            new_text = self.tokenizer.decode(new_text_ids[0], skip_special_tokens=True)
        else:
            raise NotImplementedError("model type not supported")


    def lsh_reject_completion(self, prompt: str, margin=0.002, **kwargs):
        # TODO: Move this logic to a better place and remove the function.
        if "Mixtral" in self.model.config._name_or_path:
            prompt = self.mixtral_format_instructions(prompt)
                
        sent_end_criteria = SentenceEndCriteria(self.tokenizer)
        sent_end_criteria.update(prompt)

        lsh_seed = self.lsh_model.get_hash([prompt])[0]
        accept_mask = get_mask_from_seed(self.cfg.watermark_args.sp_dim, self.cfg.watermark_args.lmbd, lsh_seed)

        text = prompt
        new_text = prompt

        text_ids = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        prompt_length = len(text_ids[0])

        total_trials = 0
        success_trials = 0  # also include trials that maxed out MAX_TRIALS
        current_trials = 0
        maxedout_trials = 0
        debug_text_segments = [(prompt, text_ids.size(1), lsh_seed)]
        
        while True:
            stopping_criteria = StoppingCriteriaList([sent_end_criteria])



            if new_text == '':
                log.info('WARNING: stopped generation because generated nothing (after discarding last generated token)')
                break

            log.info(f"Candidate text: {new_text}")
            log.info(f"Accept Mask: {accept_mask}")

            total_trials += 1
            current_trials += 1

            # Use the LSH model to reject generations too close to the margin.
            accepted_text, _ = reject_close_generation(
                    self.lsh_model, [new_text], margin=margin, cutoff=None)
            
            # If no text is accepted and the current number of trials is less than the maximum allowed trials, continue the loop.
            if (len(accepted_text) == 0 and current_trials < self.cfg.watermark_args.max_trials):
                continue
            
            # Get the LSH hash for the generated text, continue if it's not in the correct place
            lsh_candidate = self.lsh_model.get_hash([new_text])[0]

            log.info(f"LSH Candidate: {lsh_candidate}")

            if lsh_candidate not in accept_mask:
                log.info(f"Candidate text is doesn't fall into the correct place in the embedding space.")
                continue
            else:
                log.info("Candidate text falls within the semantic partition.")
            
            if (lsh_candidate in accept_mask) or current_trials >= self.cfg.watermark_args.max_trials:
                if current_trials >= self.cfg.watermark_args.max_trials:
                    log.info(
                        f'WARNING: desired semantic signature can\'t be sampled after max_trials {self.cfg.watermark_args.max_trials}')
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

    def generate_watermarked_outputs(self, prompt):
        if self.cfg.watermark_args.sp_mode == "lsh":
            return self._lsh_generate_watermarked_outputs(prompt)

        # TODO: Implement k-SemStamp.

        raise NotImplementedError
    
    def _lsh_generate_watermarked_outputs(self,prompt):

        # TODO: Why do we need to pass the length as well? Can we just pass the prompt to this function?
        prompt = extract_prompt_from_text(prompt, self.cfg.watermark_args.len_prompt)
        response = lsh_reject_completion(
            prompt,
            self.model, self.tokenizer, self.gen_config,
            self.lsh_model, self.cfg.watermark_args.sp_dim,
            lmbd=self.cfg.watermark_args.lmbd,
            device=self.cfg.watermark_args.device,
            margin=self.cfg.watermark_args.delta,
            pipeline=self.pipeline,
            generator_kwargs=self.generator_kwargs,
            max_new_tokens=self.cfg.generator_args.max_new_tokens)
        
        # TODO: This returns a tuple. It should just return the watermarked text.
        log.info(f"Prompt: {prompt}")
        log.info(f"Response: {response}")

        generated_text = response[0].strip()
        return generated_text


    def detect(self, completion):
        if self.cfg.watermark_args.sp_mode == "lsh":
            return self._lsh_detect(completion)

        # TODO: Implement k-SemStamp.

        raise NotImplementedError

    # TODO: Implement.
    def _lsh_detect(self,completion):
        sents = sent_tokenize(completion)
        z_score = detect_lsh(sents=sents, lsh_model=self.lsh_model,
                                 lmbd=self.cfg.watermark_args.lmbd, lsh_dim=self.cfg.watermark_args.sp_dim)
        
        is_detected = z_score >= self.cfg.watermark_args.z_threshold
        return is_detected, z_score
