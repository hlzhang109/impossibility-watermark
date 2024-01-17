from generate import TextGenerator
from watermark import Watermark
from oracle import Oracle
from mutate import TextMutator


class AttackTester:
    def __init__(self, args):
        self.watermarking_scheme = Watermark(watermark_args)
    
        self.text_generation_model = TextGenerator(generator_args)
        
        self.quality_oracle = Oracle(oracle_args)
        
        self.mutator = TextMutator(config.mutator_args)
        
    def attack(self, num_trials, max_steps):
        
        
        
        
        
        