from textdiversity import (
    TokenSemantics, DocumentSemantics, AMR, # semantics
    DependencyParse, ConstituencyParse,     # syntactical
    PartOfSpeechSequence,                   # morphological
    Rhythmic                                # phonological
)
from lexical_diversity import lex_div as ld
from nltk import ngrams
from nltk.tokenize import word_tokenize
import pandas as pd

class DiversityOracle:
    def __init__(self, metrics: dict = {}, verbose=False):
        self.metrics = metrics
        self.verbose = verbose

        if not metrics:
            if self.verbose:
                print("Initializing default metrics...")
            ldhelper = LDHelper()
            unhelper = UniqueNgramHelper()
            config = {"normalize": False}
            self.metrics = {
                'TokenSemantics': TokenSemantics(config), 
                'DocumentSemantics': DocumentSemantics(config), 
                # 'AMR': AMR(config),
                'DependencyParse': DependencyParse(config), 
                'ConstituencyParse': ConstituencyParse(config),
                'PartOfSpeechSequence': PartOfSpeechSequence(config),
                # 'Rhythmic': Rhythmic(config),
                'ttr': ldhelper.ttr,
                'log_ttr': ldhelper.log_ttr,
                'root_ttr': ldhelper.root_ttr,
                'maas_ttr': ldhelper.maas_ttr,
                'mattr': ldhelper.mattr,
                'msttr': ldhelper.msttr,
                'hdd': ldhelper.hdd,
                'mtld': ldhelper.mtld,
                'mtld_ma_bid': ldhelper.mtld_ma_bid,
                'mtld_ma_wrap': ldhelper.mtld_ma_wrap,
                'unigrams': unhelper.unigrams,
                'bigrams': unhelper.bigrams,
                'trigrams': unhelper.trigrams,
            }

    def __call__(self, corpus):
        results = []
        for metric_name, metric_fn in self.metrics.items():
            if self.verbose:
                print(f"Evaluating {metric_name}...")
            try:
                diversity_score = metric_fn(corpus)
            except Exception as e:
                print(e)
                diversity_score = -1
            results.append({
                "metric_name": metric_name,
                "diversity_score": diversity_score
            })
        return results

    def compare(self, corpus1, corpus2):
        d1 = pd.DataFrame(self(corpus1))
        d2 = pd.DataFrame(self(corpus2))
        df = pd.merge(d1, d2, on="metric_name").rename(
            columns={"diversity_score_x": "corpus1_diversity_scores", 
                     "diversity_score_y": "corpus2_diversity_scores"})
        return df

class LDHelper:

    def _flemmatize(self, corpus):
        flemmas = []
        for doc in corpus:
            flemmas.extend(ld.flemmatize(doc))
        return flemmas

    def ttr(self, coprus):
        return ld.ttr(self._flemmatize(coprus))

    def root_ttr(self, coprus):
        return ld.root_ttr(self._flemmatize(coprus))

    def log_ttr(self, coprus):
        return ld.log_ttr(self._flemmatize(coprus))

    def maas_ttr(self, coprus):
        return ld.maas_ttr(self._flemmatize(coprus))

    def msttr(self, coprus):
        return ld.msttr(self._flemmatize(coprus))

    def mattr(self, coprus):
        return ld.mattr(self._flemmatize(coprus))

    def hdd(self, coprus):
        return ld.hdd(self._flemmatize(coprus))

    def mtld(self, coprus):
        return ld.mtld(self._flemmatize(coprus))

    def mtld_ma_wrap(self, coprus):
        return ld.mtld_ma_wrap(self._flemmatize(coprus))

    def mtld_ma_bid(self, coprus):
        return ld.mtld_ma_bid(self._flemmatize(coprus))


class UniqueNgramHelper:

    def _tokenize(self, corpus):
        tokens = []
        for doc in corpus:
            tokens.extend(word_tokenize(doc))
        return tokens

    def _make_unique(self, n_gram_generator):
        return len(set(list(n_gram_generator)))

    def unigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 1)
        return self._make_unique(n_gram_generator)

    def bigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 2)
        return self._make_unique(n_gram_generator)

    def trigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 3)
        return self._make_unique(n_gram_generator)
    
if __name__ == "__main__":
    
    from datasets import load_dataset

    dataset = load_dataset("chansung/llama2-stories", split="train")
    texts = [s for s in dataset["story"] if s.strip() != "" and type(s) == str]
    
    mid_point = len(texts) // 2
    max_stories = 10

    corpus1 = texts[:mid_point][:max_stories]
    corpus2 = texts[mid_point:][:max_stories]

    div_oracle = DiversityOracle(verbose=True)

    # d_scores1 = div_oracle(corpus1)
    # d_scores2 = div_oracle(corpus2)

    # print(f"corpus1: {corpus1}")
    # print(pd.DataFrame(d_scores1))
    
    # print(f"corpus2: {corpus2}")
    # print(pd.DataFrame(d_scores2))

    df = div_oracle.compare(corpus1, corpus2)

    print(df)