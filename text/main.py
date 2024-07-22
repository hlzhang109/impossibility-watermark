import torch
from evaluation.dataset import C4Dataset, HumanEvalDataset, WMT16DE_ENDataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM, AutoModelForSeq2SeqLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, RandomWalkAttack
from evaluation.tools.oracle import QualityOracle
import fire
import os

import logging
log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def get_attack(attack_name, model=None, tokenizer=None, device="cuda"):
    if attack_name == 'Word-D':
        attack = WordDeletion(ratio=0.3)
    elif attack_name == 'Word-S':
        attack = SynonymSubstitution(ratio=0.5)
    elif attack_name == 'Word-S(Context)':
        attack = ContextAwareSynonymSubstitution(ratio=0.5,
                                                 tokenizer=BertTokenizer.from_pretrained('bert-large-uncased'),
                                                 model=BertForMaskedLM.from_pretrained('bert-large-uncased').to(device))
    elif attack_name == 'Doc-P(GPT-3.5)':
        attack = GPTParaphraser(openai_model='gpt-3.5-turbo',
                                prompt='Please rewrite the following text: ')
    elif attack_name == 'Doc-P(Dipper)':
        attack = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('google/t5-v1_1-xxl'),
                                   model=T5ForConditionalGeneration.from_pretrained('kalpeshk2011/dipper-paraphraser-xxl', device_map='auto'),
                                   lex_diversity=60, order_diversity=0, sent_interval=1, 
                                   max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
    elif attack_name == 'Random-Walk':
        perturbation_model = "google/t5-v1_1-xl" 
        perturbation_oracle = AutoModelForSeq2SeqLM.from_pretrained(perturbation_model, device_map='auto')
        perturbation_tokenizer = AutoTokenizer.from_pretrained(perturbation_model)
        quality_oracle = QualityOracle(tokenizer, model, choice_granularity=5, device=device, check_quality='checker')
        span_len = 6 
        attack = RandomWalkAttack(perturbation_tokenizer=perturbation_tokenizer, perturbation_oracle=perturbation_oracle,
                                  quality_oracle=quality_oracle,
                                  span_len=span_len,
                                  max_new_tokens=int(2*span_len),  min_length=int(span_len),
                                  do_sample=True, top_p=0.8, top_k=40, repetition_penalty=1.3, temperature=1.5) 
    else:
        raise ValueError(f'Attack {attack_name} not found')
    return attack

def get_watermark(model, tokenizer, wtmk_name, max_new_tokens, temp=0.8, device="cuda"):
    gen_kwargs = {'num_return_sequences': 1, 'temperature': temp, 'top_k': 50, 'top_p': 0.95}
    print('gen_kwargs: ', gen_kwargs)
    min_length = 25 if max_new_tokens-100 <= 0 else max_new_tokens-100
    transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=tokenizer.vocab_size, #50272,
                                            device=device,
                                            max_new_tokens=max_new_tokens,
                                            min_length=min_length,
                                            do_sample=True,
                                            **gen_kwargs)
    # Load watermark algorithm
    watermark = AutoWatermark.load(wtmk_name, 
                                    algorithm_config=f'config/{wtmk_name}.json',
                                    transformers_config=transformers_config)
    return watermark

def main(model_name_or_path, wtmk_name, attack_name, max_new_tokens, 
         num_samples=400, data_name='c4', save_folder='./results', device='cuda'):
    if data_name == 'c4':
        dataset = C4Dataset(f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_unwatermarked.json', num_samples=num_samples, watermark_source=f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_watermarked.json')
    else:
        raise ValueError(f'Dataset {data_name} not found')
    
    print(f"max_new_tokens: {max_new_tokens}")
    print(f'Watermark: {wtmk_name}')
    print(f'model_name_or_path: {model_name_or_path}')
    print(f'Attack: {attack_name}')
    print(f'save_folder: {save_folder}')
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    watermark = get_watermark(model, tokenizer, wtmk_name, max_new_tokens=max_new_tokens, device=device)
    attack = get_attack(attack_name, model, tokenizer)

    attack_pipe = WatermarkedTextDetectionPipeline(dataset=dataset, text_editor_list=[attack], # TruncatePromptTextEditor(),
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 
    wtmk_pipe = WatermarkedTextDetectionPipeline(dataset=dataset, text_editor_list=[TruncatePromptTextEditor()],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)
    
    if os.path.exists(f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_unwatermarked.pt'):
        unwtmk_pt = torch.load(f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_unwatermarked.pt')
        unwatermarked_results = unwtmk_pt.tolist()
        print("Loaded exsiting unwatermarked results")
    else:
        unwtmk_pipe = UnWatermarkedTextDetectionPipeline(dataset=dataset, text_editor_list=[],
                                                    show_progress=True, return_type=DetectionPipelineReturnType.SCORES)
        unwatermarked_results = unwtmk_pipe.evaluate(watermark)
        print('Unwatermarked detection results:')
        print(unwatermarked_results)
        unwtmk_pt = torch.tensor(unwatermarked_results)
        torch.save(unwtmk_pt, f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_unwatermarked.pt')
    
    if os.path.exists(f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_watermarked.pt'):
        wtmk_pt = torch.load(f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_watermarked.pt')
        watermarked_results = wtmk_pt.tolist()
        print("Loaded exsiting watermarked results")
    else:
        watermarked_results = wtmk_pipe.evaluate(watermark)
        print('Watermarked detection results:')
        print(watermarked_results)
        wtmk_pt = torch.tensor(watermarked_results)
        torch.save(wtmk_pt, f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_watermarked.pt')
    calculator = DynamicThresholdSuccessRateCalculator(labels=['FPR', 'TPR', 'F1', 'P', 'R', 'ACC'], rule='best')
    print(calculator.calculate(watermarked_results, unwatermarked_results))

    attack_results = attack_pipe.evaluate(watermark, attack_name=attack_name)
    print('attack detection results:')
    print(attack_results)
    attack_pt = torch.tensor(attack_results)
    torch.save(attack_pt, f'{save_folder}/{wtmk_name}_{max_new_tokens}_{data_name}_{attack_name}.pt')
    calculator = DynamicThresholdSuccessRateCalculator(labels=['FPR', 'TPR', 'F1', 'P', 'R', 'ACC'], rule='best')
    print(calculator.calculate(attack_results, unwatermarked_results))

if __name__ == '__main__':
    fire.Fire(main)