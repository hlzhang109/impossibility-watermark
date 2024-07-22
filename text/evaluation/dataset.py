# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json
import os

class BaseDataset:
    """Base class for dataset."""

    def __init__(self, num_samples: int = 400, watermark_source: str = '', eval_attack=False):
        """Initialize the dataset."""
        self.prompts = []
        self.natural_texts = []
        self.references = []
        self.num_samples = num_samples
        self.watermark_source = watermark_source

        self.watermarked_texts = []
        if watermark_source and os.path.exists(watermark_source):
            with open(watermark_source, 'r') as f:
                lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                if eval_attack:
                    self.watermarked_texts.append(item['attacked_text'])
                else:
                    self.watermarked_texts.append(item['watermarked_text'])

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_watermarked_text(self, index):
        """Return the prompt at the specified index."""
        return self.watermarked_texts[index]

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, num_samples: int = 400, watermark_source: str = '', eval_attack = False):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source, eval_attack=eval_attack)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
            print(self.data_source)
            lines = f.readlines()
        for line in lines[:self.num_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])


class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(self, data_source: str, num_samples: int = 400, watermark_source: str = '') -> None:
        """
            Initialize the WMT16 DE-EN dataset.

            Parameters:
                data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:200]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(self, data_source: str, num_samples: int = 400, watermark_source: str = '') -> None:
        """
            Initialize the HumanEval dataset.

            Parameters:
                data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:100]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})


if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json')
    d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl')
