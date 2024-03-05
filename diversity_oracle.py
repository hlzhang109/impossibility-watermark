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

def get_metrics(metrics, normalized):
    if normalized:
        normalized_config = {"normalize": True, "split_sentences": True}
        normalized_unhelper = UniqueNgramHelper(normalized_config)
        
        metrics = {
            'normalized_TokenSemantics' : TokenSemantics(normalized_config),
            'normalized_DependencyParse': DependencyParse(normalized_config), 
            'normalized_ConstituencyParse': ConstituencyParse(normalized_config),
            'normalized_PartOfSpeechSequence': PartOfSpeechSequence(normalized_config),   
            'normalized_unique_unigrams': normalized_unhelper.unigrams,
            'normalized_unique_bigrams': normalized_unhelper.bigrams,
            'normalized_unique_trigrams': normalized_unhelper.trigrams,
        }        
    else:
        metrics = {
            'TokenSemantics': TokenSemantics(config), 
            # 'DocumentSemantics': DocumentSemantics(config), 
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
            'unique_unigrams': unhelper.unigrams,
            'unique_bigrams': unhelper.bigrams,
            'unique_trigrams': unhelper.trigrams,
        }        

class DiversityOracle:
    def __init__(self, metrics: dict = {}, verbose=False, normalized=True):
        self.metrics = metrics
        self.verbose = verbose

        if not metrics:
            if self.verbose:
                print("Initializing default metrics...")
                
            config = {"normalize": False, "split_sentences": True}
            unhelper = UniqueNgramHelper(config)
            
            ldhelper = LDHelper()
            
            self.metrics = {
                'TokenSemantics': TokenSemantics(config), 
                'normalized_TokenSemantics' : TokenSemantics(normalized_config),
                # 'DocumentSemantics': DocumentSemantics(config), 
                # 'AMR': AMR(config),
                'DependencyParse': DependencyParse(config), 
                'ConstituencyParse': ConstituencyParse(config),
                'PartOfSpeechSequence': PartOfSpeechSequence(config),
                'normalized_DependencyParse': DependencyParse(normalized_config), 
                'normalized_ConstituencyParse': ConstituencyParse(normalized_config),
                'normalized_PartOfSpeechSequence': PartOfSpeechSequence(normalized_config),
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
                'unique_unigrams': unhelper.unigrams,
                'unique_bigrams': unhelper.bigrams,
                'unique_trigrams': unhelper.trigrams,                
                'normalized_unique_unigrams': normalized_unhelper.unigrams,
                'normalized_unique_bigrams': normalized_unhelper.bigrams,
                'normalized_unique_trigrams': normalized_unhelper.trigrams,
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

    default_config = {
        'normalize': False,
    }

    def __init__(self, config={}):
        self.config = {**self.default_config, **config} 

    def _tokenize(self, corpus):
        tokens = []
        for doc in corpus:
            tokens.extend(word_tokenize(doc))
        return tokens

    def unigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_grams = list(ngrams(tokens, 1))
        unique_n_grams = set(n_grams)
        if self.config["normalize"]:
            return len(unique_n_grams) / len(n_grams)
        return len(unique_n_grams)

    def bigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_grams = list(ngrams(tokens, 2))
        unique_n_grams = set(n_grams)
        if self.config["normalize"]:
            return len(unique_n_grams) / len(n_grams)
        return len(unique_n_grams)

    def trigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_grams = list(ngrams(tokens, 3))
        unique_n_grams = set(n_grams)
        if self.config["normalize"]:
            return len(unique_n_grams) / len(n_grams)
        return len(unique_n_grams)
    
if __name__ == "__main__":
    
    # from datasets import load_dataset

    # dataset = load_dataset("chansung/llama2-stories", split="train")
    # texts = [s for s in dataset["story"] if s.strip() != "" and type(s) == str]
    
    # mid_point = len(texts) // 2
    # max_stories = 10

    # corpus1 = texts[:mid_point][:max_stories]
    # corpus2 = texts[mid_point:][:max_stories]
    
    corpus1_path = './inputs/lotr_gollum_watermarked_attacks_1.csv'
    corpus2_path = './inputs/lotr_gollum_watermarked_attacks_3.csv'
    
    corpus1 = pd.read_csv(corpus1_path)['text'].tolist()
    corpus2 = pd.read_csv(corpus2_path)['text'].tolist()
    
    div_oracle = DiversityOracle(verbose=True)

    d_scores1 = div_oracle(corpus1)
    
    print(d_scores1)
    print(type(d_scores1))
    # d_scores2 = div_oracle(corpus2)

    # print(f"corpus1: {corpus1}")
    # print(pd.DataFrame(d_scores1))
    
    # print(f"corpus2: {corpus2}")
    # print(pd.DataFrame(d_scores2))

    # df = div_oracle.compare(corpus1, corpus2)

    # print(df)