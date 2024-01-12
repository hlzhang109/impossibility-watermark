import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, AutoConfig

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def main():
    print("Starting the script...")

    # model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading the model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)
    # config.quantization_config = {"disable_exllama": True}

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map="auto",
                                            trust_remote_code=False,
                                            revision="main",
                                            config=config)

    # Load the dataset
    print("Loading the C4 RealNews dataset subset...")
    c4_realnews_subset = load_dataset("c4", "realnewslike", split='train', streaming=True, trust_remote_code=True)

    first_10_stories = []

    for story in c4_realnews_subset.take(50):
        first_10_stories.append(story)

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                delta=2.0,
                                                seeding_scheme="selfhash")

    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=0.25,
                                            seeding_scheme="selfhash",
                                            device=model.device,
                                            tokenizer=tokenizer,
                                            z_threshold=4.0,
                                            normalizers=[],
                                            ignore_repeated_ngrams=True)

    # Prepare for CSV output
    data_for_json = []
    
    successful_completions = 0
    successful_watermark_detections = 0

    for i, entry in enumerate(first_10_stories):
        if i % 100 == 0:
            print(f"Processing entry {i}...")

        # Extract the first 20 tokens from the original text
        prefix = " ".join(entry['text'].split()[:20])

        # Tokenize and generate completion
        inputs = tokenizer(prefix, return_tensors="pt", padding=True, truncation=True, max_length=30)
        inputs = inputs.to(model.device)

        outputs = model.generate(**inputs, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=200, logits_processor=LogitsProcessorList([watermark_processor]), repetition_penalty = 1.2)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        score = watermark_detector.detect(completion)

        print(completion)

        score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}

        entry = {
            'Prefix': prefix,
            'Completion': completion,
            'Score': score_dict
        }
        
        if score_dict['num_tokens_scored'] > 60:
            successful_completions += 1
        if score_dict['prediction']:
            successful_watermark_detections += 1
        

        # Store the prefix and completion
        data_for_json.append(entry)
        
        

    # Write the prefixes and completions to a CSV file
    print("Writing data to JSON file...")
    filename = f"text_completions_{time.time()}.json"
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_for_json, file, ensure_ascii=False, indent=4)

    print("Completions saved to text_completions.json")

    # Give average statistics.    
    
    print(f"Successful completions: {successful_completions}")
    print(f"Successful watermarked detections: {successful_watermark_detections}")

if __name__ == "__main__":
    main()