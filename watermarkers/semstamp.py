import logging

import torch
import nltk
import os
from sentence_transformers import SentenceTransformer
from custom_transformer import CustomSentenceTransformer
from transformers import GenerationConfig, StoppingCriteriaList, AutoModel
from nltk.tokenize import sent_tokenize
import textwrap

from watermarker import Watermarker
from watermarkers.SemStamp.sbert_lsh_model import SBERTLSHModel
from watermarkers.SemStamp.sampling_lsh_utils import get_mask_from_seed, reject_close_generation
from watermarkers.SemStamp.sampling_utils import extract_prompt_from_text, SentenceEndCriteria, gen_sent, discard_final_token_in_outputs
from watermarkers.SemStamp.detection_utils import detect_lsh
from watermarkers.SemStamp.sampling_kmeans_utils import embed_gen_list, get_cluster_centers, load_embeds, kmeans_reject_overlap, get_cluster_id, get_cluster_mask
from utils import mixtral_format_instructions, parse_llama_output, save_to_csv_with_filepath, replace_multiple_commas

nltk.download('punkt')

PUNCTS = '.,!?'

log = logging.getLogger(__name__)

# TODO: Throws a circular import errors when in utils.py, we eventually want to fix this.
def list_to_comma_separated_string(int_list):
    """
    Converts a list of integers to a comma-separated string.

    Args:
    int_list (list of int): The list of integers to convert.

    Returns:
    str: A comma-separated string of integers.
    """
    return ','.join(map(str, int_list))

class SemStampWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, **kwargs):
        super().__init__(cfg, pipeline, n_attempts, **kwargs)
        self.gen_config = None

    def _setup_generating_components(self):
        """
        This function sets up the LLM we'll use for generating watermarked text.
        """
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

        self.model.eval()

        log.info(self.generator_kwargs)

    def _setup_watermark_components(self):
        # NOTE: currently, no batching.
        # We don't want to initialize Llama when we only want to detect.
        if not self.only_detect:
            log.info("Setting up generating components...")
            self._setup_generating_components()

        log.info(f"Initializing embedder model.")

        if self.cfg.watermark_args.use_fine_tuned:
            log.info(f"Using the fine-tuned SentenceTransformer...")
            self.embedder = CustomSentenceTransformer(model_name="AbeHou/SemStamp-c4-sbert", device=self.device)
        else:
            log.info(f"Using the generic SentenceTransformer...")
            self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")
            self.embedder.eval()

        log.info(f"Finished initializing embedder model.")
        
        if self.cfg.watermark_args.sp_mode == "lsh":
            self.lsh_model = SBERTLSHModel(lsh_model_path=None,
                                  device=self.cfg.watermark_args.device, batch_size=1, lsh_dim=self.cfg.watermark_args.sp_dim, sbert_type='base', embedder=self.embedder)
        
    def generate_sentence(self, text, text_ids, stopping_criteria):
        if "opt" in self.model.config._name_or_path:
            candidate_text, candidate_text_ids = gen_sent(model = self.model, 
                tokenizer = self.tokenizer, 
                text_ids = text_ids,
                gen_config = self.gen_config,
                stopping_criteria = stopping_criteria
            )
        elif "Mixtral" in self.model.config._name_or_path:
            self.generator_kwargs['stopping_criteria'] = stopping_criteria
            outputs = self.model.generate(text_ids, self.gen_config, **self.generator_kwargs)
            log.info(f"Outputs: {outputs}")
            outputs = discard_final_token_in_outputs(outputs)
            candidate_text_ids = outputs.sequences
            candidate_text = self.tokenizer.decode(candidate_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
        elif "Llama" in self.model.config._name_or_path:
            self.generator_kwargs['stopping_criteria'] = stopping_criteria
            outputs = self.model.generate(inputs=text_ids, generation_config=self.gen_config, **self.generator_kwargs)
            outputs = outputs[:, :-1]  # Remove the last token from each sequence
            candidate_text_ids = outputs
            candidate_text = self.tokenizer.decode(candidate_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
            candidate_text = replace_multiple_commas(candidate_text)
        else:
            raise NotImplementedError("model type not supported")
        
        return candidate_text, candidate_text_ids

    def generate_watermarked_outputs(self, prompt):
        if self.cfg.watermark_args.sp_mode == "lsh":
            return self._lsh_generate_watermarked_outputs(prompt)
        if self.cfg.watermark_args.sp_mode == "kmeans":
            return self._kmeans_generate_watermarked_outputs(prompt)
        raise NotImplementedError
    
    def _lsh_reject_completion(self, prompt: str, margin=0.002, stats_csv_path=None, **kwargs):
        # TODO: Can we move this logic to a better place?
        # We don't want to make it a prompt if it's not a completion.
        if not self.cfg.is_completion:
            if "Mixtral" in self.model.config._name_or_path:
                prompt = mixtral_format_instructions(prompt)
            if "Llama" in self.model.config._name_or_path:
                prompt = textwrap.dedent(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful personal assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
        else:
            log.info(f"Generating a completion, not a prompt-based generation.")

        sent_end_criteria = SentenceEndCriteria(self.tokenizer)
        sent_end_criteria.update(prompt)

        lsh_seed = self.lsh_model.get_hash([prompt])[0]
        accept_mask = get_mask_from_seed(self.cfg.watermark_args.sp_dim, self.cfg.watermark_args.lmbd, lsh_seed)

        text = prompt
        candidate_text = prompt

        text_ids = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        prompt_length = len(text_ids[0])

        total_sentences = 0
        successful_sentences = 0
        maxed_out_sentences = 0

        debug_text_segments = [(prompt, text_ids.size(1), lsh_seed)]

        current_num_tries = 0 # how many times how we tried to generate the current sentence
        
        while True:
            stopping_criteria = StoppingCriteriaList([sent_end_criteria])

            candidate_text, candidate_text_ids = self.generate_sentence(text, text_ids, stopping_criteria)

            if candidate_text == '':
                log.info('WARNING: stopped generation because generated nothing (after discarding last generated token)')
                break
            
            log.info(f"Candidate text: {candidate_text}")
            log.info(f"Accept Mask: {accept_mask}")

            total_sentences += 1
            current_num_tries += 1

            # Use the LSH model to reject generations too close to the margin.
            accepted_text, _ = reject_close_generation(
                    self.lsh_model, [candidate_text], margin=margin, cutoff=None)
            
            # If no text is accepted and the current number of trials is less than the maximum allowed trials, continue the loop.
            if (len(accepted_text) == 0 and current_num_tries < self.cfg.watermark_args.max_trials):
                # TODO: Eventually want to pass in a directory and file name, just getting a quick and dirty implementation right now
                # so we can run attacks.
                # TODO: Turn these into a single function call.
                if stats_csv_path is not None:
                    accept_mask_list= accept_mask.cpu().tolist() # needs to be on CPU
                    # log.info(f"acceptmasklist: {accept_mask_list}")

                    accept_mask_str = list_to_comma_separated_string(accept_mask_list)
                    # log.info(f"acceptmaskstr: {accept_mask_str}")

                    stats = {
                        "total_sentences" : total_sentences,
                        "candidate_text" : candidate_text,
                        "passed_margin_test" : False,
                        "candidate_text_lsh" : None,
                        "accept_mask" : None,
                        "one_sentence" : None,
                        "current_num_tries" : current_num_tries,
                    }
                    save_to_csv_with_filepath([stats], stats_csv_path)
                

                continue
            
            # Get the LSH hash for the generated text, continue if it's not in the correct place
            lsh_candidate = self.lsh_model.get_hash([candidate_text])[0]

            log.info(f"LSH Candidate: {lsh_candidate}")

            one_sentence = len(sent_tokenize(candidate_text)) == 1
            candidate_accepted = (lsh_candidate in accept_mask) and one_sentence

            if lsh_candidate not in accept_mask:
                log.info(f"Candidate text is doesn't fall into the correct place in the embedding space.")
            # NOTE: I don't know why, but Mixtral seemed to generate 2 sentences 10% of the time. This is meant to avoid the issue.
            elif len(sent_tokenize(candidate_text)) > 1:
                log.info(f"Candidate text is more than one sentence.")
            else:
                log.info("Candidate text falls within the semantic partition.")

            # logging for analyzing generation stats
            # TODO: Eventually want to pass in a directory and file name, just getting a quick and dirty implementation right now
            # so we can run attacks.
            if stats_csv_path is not None:
                accept_mask_list= accept_mask.cpu().tolist() # needs to be on CPU
                log.info(f"acceptmasklist: {accept_mask_list}")

                accept_mask_str = list_to_comma_separated_string(accept_mask_list)
                log.info(f"acceptmaskstr: {accept_mask_str}")

                stats = {
                    "total_sentences" : total_sentences,
                    "candidate_text" : candidate_text,
                    "passed_margin_test" : True,
                    "candidate_text_lsh" : lsh_candidate,
                    "accept_mask" : accept_mask_str,
                    "one_sentence" : one_sentence,
                    "current_num_tries" : current_num_tries,
                }
                save_to_csv_with_filepath([stats], stats_csv_path)

            if candidate_accepted or current_num_tries >= self.cfg.watermark_args.max_trials:
                if current_num_tries >= self.cfg.watermark_args.max_trials:
                    log.info(
                        f'WARNING: desired semantic signature can\'t be sampled after max_trials {self.cfg.watermark_args.max_trials}')
                    log.info(f'CONTEXT: {text}')
                    log.info(
                        f'NOTE: use regular (non-filtered-by-sig) continuation: {candidate_text}')
                    maxed_out_sentences += 1

                debug_text_segments.append(
                    (candidate_text, candidate_text_ids.size(1) - text_ids.size(1), lsh_candidate))
                
                if candidate_accepted:
                    successful_sentences += 1

                # Proceed to the next sentence
                lsh_seed = lsh_candidate
                accept_mask = get_mask_from_seed(self.cfg.watermark_args.sp_dim, self.cfg.watermark_args.lmbd, lsh_seed)
                text += candidate_text
                text_ids = candidate_text_ids
                sent_end_criteria.update(text)
                current_num_tries = 0

                # If the number of new tokens generated reaches the maximum new token number, stop generating.
                if (len(text_ids[0]) - prompt_length) >= self.cfg.watermark_args.max_new_tokens-1:
                    break

        text = parse_llama_output(text)
        return text, total_sentences
    
    def _lsh_generate_watermarked_outputs(self,prompt):
        # If it's a completion, only use the first len_prompt many tokens.
        if self.cfg.is_completion:
            log.info(f"Since this is a completion, extracting the prompt from the original text.")
            prompt = extract_prompt_from_text(prompt, self.cfg.watermark_args.len_prompt)

        log.info(f"Passing the following prompt to the LSH reject completion function:\n {prompt}")
        response = self._lsh_reject_completion(prompt, stats_csv_path = self.cfg.generation_stats_file_path)
        
        log.info(f"Prompt: {prompt}")
        log.info(f"Response: {response}")

        generated_text = response[0].strip()
        return generated_text
    
    def _kmeans_reject_completion(self, prompt: str, cluster_centers: torch.Tensor, margin=0.01, **kwargs):
        # TODO: Add logging.
        sent_end_criteria = SentenceEndCriteria(self.tokenizer)
        curr_cluster_id = get_cluster_id(prompt, cluster_centers, self.embedder)
        mask = get_cluster_mask(curr_cluster_id, self.cfg.watermark_args.sp_dim, self.cfg.watermark_args.lmbd)

        text = prompt
        new_text = prompt
        text_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.cfg.watermark_args.device)
        prompt_length = len(text_ids[0])
        sent_end_criteria.update(new_text)

        total_trials = 0
        success_trials = 0  # also include trials that maxed out MAX_TRIALS
        current_trials = 0
        maxedout_trials = 0
        debug_text_segments = [(prompt, text_ids.size(1), curr_cluster_id)]
        cluster_id_sequence = [curr_cluster_id.item()]
        
        while True:
            # TODO: Use generate sentence here.
            # input_ids = tokenizer.encode(last_step_text, return_tensors='pt').to(device)
            stopping_criteria = StoppingCriteriaList([sent_end_criteria])
            new_text, new_text_ids = gen_sent(model = self.model, 
                    tokenizer = self.tokenizer, 
                    text_ids = text_ids,
                    gen_config = self.gen_config,
                    stopping_criteria = stopping_criteria
                )
            # print(f'NEW TEXT: {new_text}')
            if new_text == '':
                print('WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
                break

            total_trials += 1
            current_trials += 1

            accepted_text, curr_cluster_id = kmeans_reject_overlap(text=new_text, embedder=self.embedder, cluster_centers=cluster_centers, margin=margin)

            if (accepted_text == None and current_trials < self.cfg.watermark_args.max_trials):
                continue
            else:
                new_text = accepted_text if accepted_text != None else new_text
            cluster_id_sequence.append(curr_cluster_id.item())
            if (curr_cluster_id in mask) or current_trials >= self.cfg.watermark_args.max_trials:
                if current_trials >= self.cfg.watermark_args.max_trials:
                    print(
                        f'WARNING: desired semantic signature can\'t be sampled after max_trials {self.cfg.watermark_args.max_trials}', flush=True)
                    print(f"cluster_id_sequence: {cluster_id_sequence}")
                    print(
                        f"cluster_ids_counter: {torch.bincount(torch.tensor(cluster_id_sequence))}")
                    print(f'CONTEXT: {text}', flush=True)
                    print(
                        f'NOTE: use regular (non-filtered-by-sig) continuation: {new_text}', flush=True)
                    maxedout_trials += 1
                debug_text_segments.append(
                    (new_text, new_text_ids.size(1) - text_ids.size(1), curr_cluster_id))
                current_trials = 0
                success_trials += 1
                mask = get_cluster_mask(curr_cluster_id, self.cfg.watermark_args.sp_dim, self.cfg.watermark_args.lmbd)
                text += new_text
                text_ids = new_text_ids
                sent_end_criteria.update(text)
                if (len(text_ids[0]) - prompt_length) >= self.gen_config.max_new_tokens-1:
                    break
        return text, total_trials


    def _kmeans_generate_watermarked_outputs(self, prompt):
        # TODO: Waiting for Abe Hou to fix their Github repository.
        # cluster generations if no clusters provided
        if args.cc_path == None:
            if args.embed_path == None:
                embed_path = embed_gen_list(
                    embedder_path=args.embedder, dataset_path=args.train_data)
            else:
                embed_path = args.embed_path
            gen_embeds = load_embeds(embed_path)
            cluster_ids, cluster_centers = get_cluster_centers(gen_embeds, args.sp_dim)
            cc_path = os.path.join(args.train_data, f"cluster_{args.sp_dim}_centers.pt")
            torch.save(cluster_centers, cc_path)
        # load cluster centers
        else:
            cluster_centers = torch.load(args.cc_path)
        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            response= kmeans_reject_completion(
                prompt=prompt,
                cluster_centers=cluster_centers,
                margin=args.delta,
                device=args.device)
            ex['text'] = response.strip()
            return ex

    def detect(self, completion):
        if self.cfg.watermark_args.sp_mode == "lsh":
            return self._lsh_detect(completion)

        # TODO: Implement k-SemStamp detection.

        raise NotImplementedError

    def _lsh_detect(self,completion):
        sents = sent_tokenize(completion)
        z_score = detect_lsh(sents=sents, lsh_model=self.lsh_model,
                                 lmbd=self.cfg.watermark_args.lmbd, lsh_dim=self.cfg.watermark_args.sp_dim)
        
        is_detected = z_score >= self.cfg.watermark_args.z_threshold
        return is_detected, z_score
