import logging
from watermarker import Watermarker

import nltk
import torch
import os
from transformers import LogitsProcessorList, AutoModelForCausalLM, GenerationConfig, AutoTokenizer, AutoConfig
from SemStamp.sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from SemStamp.sampling_lsh_utils import lsh_reject_completion
from SemStamp.sampling_utils import extract_prompt_from_text
from SemStamp.detection_utils import detect_kmeans, detect_lsh
from nltk.tokenize import sent_tokenize

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
