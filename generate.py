import argparse
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, AutoConfig

# UMD
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
# Unigram
from gptwm import GPTWatermarkLogitsWarper, GPTWatermarkDetector

def main(args):
    print("Starting the script...")
    
    if args.watermarking_scheme not in ['umd', 'unigram']:
        print(f"{args.watermarking_scheme} is not a recognized watermarking scheme.")
        return 2

    model_name = args.model_name
    watermarking_scheme = args.watermarking_scheme
    print(f"Loading the model: {model_name}")
    print(f"Watermarking Scheme: {watermarking_scheme}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map="auto",
                                            trust_remote_code=False,
                                            revision="main",
                                            config=config)

    # # Load the dataset
    # print("Loading the C4 RealNews dataset subset...")
    # c4_realnews_subset = load_dataset("c4", "realnewslike", split='train', streaming=True, trust_remote_code=True)

    # first_10_stories = []

    # for story in c4_realnews_subset.take(50):
    #     first_10_stories.append(story)
    
    first_10_stories = ["Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?"]

    is_completion = False
    
    if watermarking_scheme == "umd": 
        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=0.25,
                                                    delta=2.0,
                                                    seeding_scheme="selfhash")

        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                seeding_scheme="selfhash",
                                                device=model.device,
                                                tokenizer=tokenizer,
                                                z_threshold=args.z_threshold,
                                                normalizers=[],
                                                ignore_repeated_ngrams=True)
    elif watermarking_scheme == "unigram":
        # TODO: Make the arguments to Unigram more systematic.
        wm_key = 0
        watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=0.5,
                                                                    strength=2.0,
                                                                    vocab_size=tokenizer.vocab_size,
                                                                    watermark_key=wm_key)])
        watermark_detector =  GPTWatermarkDetector(fraction=0.5,
                                    strength=2.0,
                                    vocab_size=tokenizer.vocab_size,
                                    watermark_key=wm_key)

    # Prepare for CSV output        
    data_for_json = []
    
    successful_completions = 0
    successful_watermark_detections = 0
    
    entry = first_10_stories[0]

    for i in range(10):
        if i % 100 == 0:
            print(f"Processing entry {i}...")

        # Extract the first 20 tokens from the original text
        if is_completion:
            prefix = " ".join(entry['text'].split()[:20])

        # Tokenize and generate completion
        inputs = tokenizer(entry, return_tensors="pt", padding=True, truncation=True, max_length=100)
        inputs = inputs.to(model.device)

        if watermarking_scheme == "umd":
            outputs = model.generate(**inputs, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1024, logits_processor=LogitsProcessorList([watermark_processor]), repetition_penalty = 1.1)
        elif watermarking_scheme == "unigram":
            outputs = model.generate(**inputs, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1024, logits_processor=watermark_processor, output_scores = True)

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not is_completion:
            completion = completion.replace(entry, '', 1)  # Replace the first occurrence of 'entry' with an empty string
            
        print(completion)

        if watermarking_scheme == "umd":   
            score = watermark_detector.detect(completion)
            score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
            
            data = {
                'Prompt': entry,
                'Completion': completion,
                'Score': score_dict
            }
            
            if score_dict['num_tokens_scored'] > 60:
                successful_completions += 1
            if score_dict['prediction']:
                successful_watermark_detections += 1
        elif watermarking_scheme == "unigram":
            token_sequence = tokenizer(completion, add_special_tokens=False)['input_ids']
            print(token_sequence)
            z_score = watermark_detector.detect(token_sequence, device=model.device)
            # print(z_score)
            prediction = (z_score >= args.z_threshold)

            data = {
                'Prompt': entry,
                'Completion': completion,
                'Score': z_score,
                'Prediction' : str(prediction)
            }
            
            print(f"Length of the completion: {len(completion)}")
            if len(completion) > 750:
                successful_completions += 1
            if prediction:
                successful_watermark_detections += 1
        
        # Store the prefix and completion
        data_for_json.append(data)
        
    # Write the prefixes and completions to a CSV file
    print("Writing data to JSON file...")
    timestamp = int(time.time())
    filename = f"text_completions_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_for_json, file, ensure_ascii=False, indent=4)

    print(f"Completions saved to {filename}")

    # Give statistics.    
    print(f"Successful completions: {successful_completions}")
    print(f"Successful watermarked detections: {successful_watermark_detections}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="TheBloke/Llama-2-7b-Chat-GPTQ")
    parser.add_argument("--watermarking_scheme", type=str, default="umd")
    parser.add_argument("--z_threshold", type=float, default=4.0)
    
    args = parser.parse_args()
    
    main(args)