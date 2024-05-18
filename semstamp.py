import logging
from watermarker import Watermarker

import nltk
import torch
import os
from transformers import LogitsProcessorList, AutoModelForCausalLM, GenerationConfig, AutoTokenizer
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
        self.setup_watermark_components()

    def setup_watermark_components(self):
        # NOTE: currently, no batching.
        # NOTE: I don't understand this.
        self.is_offline = os.environ.get('TRANSFORMERS_OFFLINE') is not None and os.environ.get(
        'TRANSFORMERS_OFFLINE') == '1'

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.generator_args.model_name_or_path, local_files_only=self.is_offline)

        # block \n
        bad_words_ids = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device='cuda').tolist()

        # NOTE: They had top_p=0.96 commented out. I don't know why.
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
						local_files_only=self.is_offline
				)
        
        # TODO: Fix lsh_model_path.
        self.lsh_model = SBERTLSHModel(lsh_model_path=None,
                                  device=self.cfg.generator_args.device_map, batch_size=1, lsh_dim=self.cfg.watermarker_args.sp_dim, sbert_type='base')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.generator_args.model_name_or_path, local_files_only=self.is_offline).to(self.cfg.generator_args.device_map)
        
        self.model.eval()
        
    def generate_watermarked_outputs(self, prompt):
        if self.cfg.watermark_args.sp_mode == "lsh":
            return self._lsh_generate_watermarked_outputs(self, prompt)

        # TODO: Implement k-SemStamp.

        raise NotImplementedError
    
    def _lsh_generate_watermarked_outputs(self,ex):

        # TODO: Why do we need to pass the length as well? Can we just pass the prompt to this function?
        prompt = extract_prompt_from_text(ex['text'], self.cfg.watermark_args.len_prompt)
        response = lsh_reject_completion(
            prompt,
            self.model, self.tokenizer, self.gen_config,
            self.lsh_model, self.cfg.watermark_args.sp_dim,
            lmbd=self.cfg.watermark_args.lmbd,
            device=self.cfg.generator_args.device_map,
            margin=self.cfg.watermark_args.delta)
        
        # TODO: This returns a tuple. It should just return the watermarked text.
        log.info(f"Prompt: {prompt}")
        log.info(f"Response: {response}")

        ex['generated_text'] = response[0].strip()
        return ex


    def detect(self, completion):
        if self.cfg.watermark_args.sp_mode == "lsh":
            return self._lsh_detect(self, completion)

        # TODO: Implement k-SemStamp.

        raise NotImplementedError

    # TODO: Implement.
    def _lsh_detect(self,completion):
        sents = sent_tokenize(completion)
        z_score = detect_lsh(sents=sents, lsh_model=self.lsh_model,
                                 lmbd=self.cfg.watermark_args.lmbd, lsh_dim=self.cfg.watermark_args.sp_dim)
        
        is_detected = z_score >= self.cfg.watermark_args.z_threshold
        return is_detected
