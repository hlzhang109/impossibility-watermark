import numpy as np

from scipy.stats import binom
import torch
from transformers import AutoModelForImageClassification, BlipImageProcessor
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from PIL.Image import Image
import PIL
from tqdm import tqdm
from cv_utils import WATERMARK_BITS, MATCH_VALUES, get_watermark_match

def get_p_value(num_bits):
    # Simulated p-value calculation, you need real statistical data to refine this. Assuming a binomial distribution under null hypothesis
    n = len(WATERMARK_BITS)
    p = 0.5  # Chance of a bit matching by chance
    p_value = 1 - binom.cdf(num_bits-1, n, p)
    return p_value

class Detector():
    def __init__(self, scheme="stable-signature"):
        self.scheme = scheme
        if scheme == "stable-signature":
            self.image_processor = BlipImageProcessor.from_pretrained("imatag/stable-signature-bzh-detector-resnet18")
            self.model = AutoModelForImageClassification.from_pretrained("imatag/stable-signature-bzh-detector-resnet18")
            self.calibration = hf_hub_download("imatag/stable-signature-bzh-detector-resnet18", filename="calibration.safetensors")
            with safe_open(self.calibration, framework="pt") as f:
                self.calibration_logits = f.get_tensor("logits")

    def detect(self, image):
        if self.scheme == "invisible-watermark":
            if isinstance(image, Image):
                image = np.array(image)
            num_bits = get_watermark_match(image)
            k = 0
            while num_bits > MATCH_VALUES[k][0]:
                k += 1
            p_value = get_p_value(num_bits)
            if p_value < 1e-3:
                print("watermark detected")
            else:
                print("no watermark detected")
            return {"p-value": p_value, "num_bits": int(num_bits), "match": MATCH_VALUES[k][1]}
        elif self.scheme == "stable-signature":
            inputs = self.image_processor(image, return_tensors="pt")
            pred = self.model(**inputs).logits[0,0] < 0.05 # 0?
            with torch.no_grad():
                output = self.model(**inputs).logits[...,0:1]
                p = (1 + torch.sum(self.calibration_logits <= output, dim=-1)) / self.calibration_logits.shape[0]
                p = p.item()
            return {"p-value": p, "num_bits": None, "match": "watermark detected" if pred else "no watermark detected"}
        else:
            raise NotImplementedError(f"Scheme {self.scheme} not implemented")

if __name__ == "__main__":
    import os
    import json
    scheme = "stable-signature" #"invisible-watermark"
    path = f"hpsv2_results/turbo/{scheme}" #f"hpsv2_results/{scheme}"
    num_samples = 200
    results = []
    for step in tqdm(range(0, 205, 5)):
        img_id = 0
        detector = Detector(scheme=scheme)
        p_value = 0
        valid_p_value = 0
        valid_indices = []

        for i in tqdm(range(num_samples)):
            filename = f"img{i}-step{step}.png"
            if step == 0:
                filename = f"img{i}-original.png"
            res = detector.detect(PIL.Image.open(f"{path}/{filename}"))
            p_value += res["p-value"]
            if res["p-value"] < 1e-3:
                valid_indices.append(i)
                valid_p_value += res["p-value"]

        res.update({"id": img_id, "step": step})
        results.append(res)
        print(res)
        with open(f"{path}/img{img_id}_stepwise_detection.jsonl", "w") as f:
            for res in results:
                f.write(json.dumps(res) + '\n')

        print(f"Average p-value: {p_value/num_samples} over {num_samples} samples")