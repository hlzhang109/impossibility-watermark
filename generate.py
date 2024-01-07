import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from extended_watermark_processor import WatermarkLogitsProcessor

def main():
    print("Starting the script...")

    model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
    print(f"Loading the model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

    # Load the dataset
    print("Loading the C4 RealNews dataset subset...")
    c4_realnews_subset = load_dataset("c4", "realnewslike", split='train[:10]')

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=0.25,
                                                   delta=2.0,
                                                   seeding_scheme="selfhash")

    # Prepare for CSV output
    data_for_csv = []

    for i, entry in enumerate(c4_realnews_subset):
        if i % 100 == 0:
            print(f"Processing entry {i}...")

        # Extract the first 20 tokens from the original text
        prefix = " ".join(entry['text'].split()[:20])

        # Tokenize and generate completion
        inputs = tokenizer(prefix, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=50, logits_processor=LogitsProcessorList([watermark_processor]))
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store the prefix and completion
        data_for_csv.append((prefix, completion))

    # Write the prefixes and completions to a CSV file
    print("Writing data to CSV file...")
    with open('./text_completions.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Prefix', 'Completion'])  # Header
        for prefix, completion in data_for_csv:
            writer.writerow([prefix, completion])

    print("Completions saved to text_completions.csv")

if __name__ == "__main__":
    main()
