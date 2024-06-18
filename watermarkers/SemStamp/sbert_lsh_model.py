# from sim_api import get_vec_para_repl
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
import os
from sentence_transformers import SentenceTransformer
from copy import deepcopy
from scipy.spatial.distance import hamming, cosine
from nearpy.hashes import RandomBinaryProjections
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from custom_transformer import CustomSentenceTransformer
from more_itertools import chunked
import numpy as np
from typing import List, Tuple, Callable, Optional, Iterator
global Device


class LSHModel:
    def __init__(self, device, batch_size, lsh_dim):
        self.comparator: Callable[[np.ndarray, np.ndarray], float]
        self.hasher = None
        self.do_lsh: bool = False
        self.dimension: int = -1
        self.device = device
        self.batch_size: int = batch_size
        self.lsh_dim: int = lsh_dim
        print("initializing random projection LSH model")
        self.hasher = RandomBinaryProjections(
            'rbp_perm', projection_count=self.lsh_dim, rand_seed=1234)
        self.do_lsh = True
        self.comparator = lambda x, y: hamming(*[
            np.fromstring(self.hasher.hash_vector(i)[0], 'u1') - ord('0')
            for i in [x, y]])
        self.comparator = lambda x, y: cosine(x, y)

    def compute_distances(self, refs: List[str], cands: List[str]) -> np.ndarray:
        '''
        :param refs: list of reference sentences
        :param cands: list of candidate sentences to compute similarity distances from references
        :return:
        '''
        assert len(refs) == len(cands)
        results = np.zeros(len(refs))
        i = 0
        for batch in chunked(zip(refs, cands), self.batch_size, total=len(refs)):
            (ref_b, cands_b) = list(zip(*batch))
            assert len(ref_b) <= self.batch_size
            [ref_features, cand_features] = [
                self.get_embeddings(x) for x in [ref_b, cands_b]]

            if i == 0:
                print(
                    f"comparing vectors of dimension {ref_features.shape[-1]}")
            results[i:i + len(ref_b)] = np.fromiter(
                map(lambda args: self.comparator(*args), zip(ref_features, cand_features)), dtype=float)
            i += len(ref_b)

        return results

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        '''
        retrieve np array of sentence embeddings from sentence iterator
        :param sents: set of sentence strings
        :return: extracted embeddings
        '''
        raise NotImplementedError()

    def get_hash(self, sents: Iterator[str]) -> Iterator[str]:
        # Get the embedding of every sentence in a matrix
        embd = self.get_embeddings(sents)
        # print(f"embedding: {embd}")
        # Get the hash of the embedding vectors in binary
        hash_strs = [self.hasher.hash_vector(e)[0] for e in embd]
        # Convert them to integers
        hash_ints = [int(s, 2) for s in hash_strs]
        return hash_ints


class SBERTLSHModel(LSHModel):
    def __init__(self, device, batch_size, lsh_dim, sbert_type='roberta', lsh_model_path=None, embedder=None, **kwargs):
        super(SBERTLSHModel, self).__init__(device, batch_size, lsh_dim)
        self.sbert_type = sbert_type
        self.dimension = 1024 if 'large' in self.sbert_type else 768

        print(f'loading SBERT {self.sbert_type} model...')

        # TODO: This should be MUCH cleaner, I just don't know how to make it cleaner. - Boran

        if embedder is not None:
            self.embedder = embedder
        else:
            if lsh_model_path == "AbeHou/SemStamp-c4-sbert":
                self.embedder = CustomSentenceTransformer(lsh_model_path, device=self.device)
            if lsh_model_path is not None:
                self.embedder = SentenceTransformer(lsh_model_path)
                self.dimension = self.embedder.get_sentence_embedding_dimension()
            else:
                self.embedder = SentenceTransformer(
                    "sentence-transformers/all-mpnet-base-v1")
                
            if lsh_model_path != "AbeHou/SemStamp-c4-sbert":
                self.embedder = self.embedder.to(self.device)
                self.embedder.eval()


        # TODO: Removing this for now since it breaks my wrapper. - Boran 
        # if lsh_model_path != "AbeHou/SemStamp-c4-sbert" and not list(self.embedder.parameters()):
        #     raise ValueError("No parameters found in model; check model path or initialization.")
        
        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        all_embeddings = self.embedder.encode(
            sents, batch_size=self.batch_size)
        return np.stack(all_embeddings)