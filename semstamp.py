import logging
from watermarker import Watermarker

import torch
import os
from transformers import LogitsProcessorList, AutoModelForCausalLM, GenerationConfig
from SemStamp.sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from SemStamp.sampling_lsh_utils import lsh_reject_completion
from SemStamp.detection_utils import detect_kmeans, detect_lsh,
# TODO: This is probably from k-SemStamp. It generates a bug right now.
# from sampling_kmeans_utils import embed_gen_list, get_cluster_centers, kmeans_reject_completion, load_embeds

# TODO: The embed issue after fixing it in their code.

log = logging.getLogger(__name__)

class SemStampWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, is_completion=False):
        super().__init__(cfg, pipeline, n_attempts, is_completion)
        self.setup_watermark_components()

    def setup_watermark_components(self):

        is_offline = os.environ.get('TRANSFORMERS_OFFLINE') is not None and os.environ.get(
        'TRANSFORMERS_OFFLINE') == '1'
        
        # block \n
        bad_words_ids = self.tokenizer(
						"\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device='cuda').tolist()
				
        # TODO: we should have a specific config file for the semstamp parameters.
        gen_config = GenerationConfig.from_pretrained(
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

        if self.cfg.watermark_args.sp_mode == "lsh":
          # TODO: What do we do with the device?
          lsh_model = SBERTLSHModel(lsh_model_path=None, #self.cfg.watermark_args.embedder,
                                    device=self.cfg.device, batch_size=1, lsh_dim=self.cfg.watermark_args.sp_dim, sbert_type='base')
          model = AutoModelForCausalLM.from_pretrained(
              self.cfg.generator_args.model_name_or_path, local_files_only=is_offline).to(self.cfg.device)
          model.eval()
          def text_to_generated_text(ex):
              prompt = ex[:self.cfg.watermark_args.len_prompt]
              response = lsh_reject_completion(
                  prompt,
                  model, self.tokenizer, self.gen_config,
                  lsh_model, self.cfg.watermark_args.sp_dim,
                  lmbd=self.cfg.watermark_args.lmbd,
                  device=self.cfg.watermark_args.device,
                  margin=self.cfg.watermark_args.delta)
              return response.strip()
        # TODO: Currently not implemented.
        elif 'kmeans' in self.cfg.watermark_args.sp_mode:
            # model = AutoModelForCausalLM.from_pretrained(
            #     self.cfg.generator_args.model_name_or_path, local_files_only=is_offline).to(self.model.device)
            # model.eval()
            # cluster generations if no clusters provided
            if self.cfg.watermark_args.cc_path == None:
                if self.cfg.watermark_args.embed_path == None:
                    embed_path = embed_gen_list(
                        embedder_path=self.cfg.watermark_args.embedder, dataset_path=self.cfg.watermark_args.train_data)
                else:
                    embed_path = self.cfg.watermark_args.embed_path
                gen_embeds = load_embeds(embed_path)
                cluster_ids, cluster_centers = get_cluster_centers(gen_embeds, self.cfg.watermark_args.sp_dim)
                cc_path = os.path.join(self.cfg.watermark_args.train_data, f"cluster_{self.cfg.watermark_args.sp_dim}_centers.pt")
                torch.save(cluster_centers, cc_path)
            # load cluster centers
            else:
                cluster_centers = torch.load(self.cfg.watermark_args.cc_path)
            embedder = SentenceTransformer(self.cfg.watermark_args.embedder, device = 'cuda')
            def text_to_generated_text(ex):
                prompt = ex[:self.cfg.watermark_args.len_prompt]
                response= kmeans_reject_completion(
                    prompt=prompt,
                    model=self.model, tokenizer=self.tokenizer, gen_config=gen_config,
                    embedder=embedder,
                    cluster_centers=cluster_centers,
                    lmbd=self.cfg.watermark_args.lmbd,
                    k_dim=self.cfg.watermark_args.sp_dim,
                    margin=self.cfg.watermark_args.delta,
                    device=self.cfg.watermark_args.device)
                return response.strip()
        else:
            raise NotImplementedError
        
        self.watermark_processor = text_to_generated_text
				# self.watermark_processor = WatermarkLogitsProcessor(
        #     vocab=list(self.tokenizer.get_vocab().values()),
        #     gamma=self.cfg.watermark_args.gamma,
        #     delta=self.cfg.watermark_args.delta,
        #     seeding_scheme=self.cfg.watermark_args.seeding_scheme
        # )
        
        # self.watermark_detector = WatermarkDetector(
        #     tokenizer=self.tokenizer,
        #     vocab=list(self.tokenizer.get_vocab().values()),
        #     z_threshold=self.cfg.watermark_args.z_threshold,
        #     gamma=self.cfg.watermark_args.gamma,
        #     seeding_scheme=self.cfg.watermark_args.seeding_scheme,
        #     normalizers=self.cfg.watermark_args.normalizers,
        #     ignore_repeated_ngrams=self.cfg.watermark_args.ignore_repeated_ngrams,
        #     device=self.cfg.watermark_args.device,
        # )
        
        # self.generator_kwargs["logits_processor"] = LogitsProcessorList([self.watermark_processor])

    def generate_watermarked_outputs(self, prompt):
        return self.watermark_processor(prompt)
        
        # inputs = self.tokenizer(
        #     prompt, 
        #     return_tensors="pt", 
        #     truncation=True, 
        #     max_length=self.cfg.generator_args.max_new_tokens
        # ).to(self.model.device)
        # outputs = self.model.generate(**inputs, **self.generator_kwargs)
        # return outputs

    def detect(self, completion):
        # ksemstamp detection
        if self.cfg.watermark_args.detection_mode == 'kmeans':
            cluster_centers = torch.load(self.cfg.watermark_args.cc_path)
            embedder = SentenceTransformer(self.cfg.watermark_args.embedder)
            z_score = detect_kmeans(sents=completion, embedder=embedder, lmbd=self.cfg.watermark_args.lmbd,
                                    k_dim=self.cfg.watermark_args.sp_dim, cluster_centers=cluster_centers)

        # semstamp detection
        elif self.cfg.watermark_args.detection_mode == 'lsh':
            lsh_model_class = SBERTLSHModel
            lsh_model = lsh_model_class(
            lsh_model_path=self.cfg.watermark_args.embedder, device='cuda', batch_size=1, lsh_dim=self.cfg.watermark_args.sp_dim, sbert_type='base')
            z_score = detect_lsh(sents=completion, lsh_model=lsh_model,
                                lmbd=self.cfg.watermark_args.lmbd, lsh_dim=self.cfg.watermark_args.sp_dim)
        return (self.cfg.watermark_args.z_threshold <= z_score), z_score
        
        # score = self.watermark_detector.detect(completion)
        # score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
        # z_score = score_dict['z_score']
        # is_detected = score_dict['prediction']
        # return is_detected, z_score