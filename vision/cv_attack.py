import os
import torch
from PIL import Image
import fire
import logging
import abc
from tqdm import tqdm
import copy
import json
from transformers import  CLIPProcessor, CLIPModel
from diffusers import StableDiffusionInpaintPipeline, LMSDiscreteScheduler
from detector import Detector
from data import WatermarkedImageDataset
from torch.utils.data import DataLoader
from cv_utils import generate_square_mask, interpolate_img

log = logging.getLogger("train")
logging.basicConfig(level=logging.INFO)

class QualityOracle(abc.ABC):
    def __init__(self, tie_threshold=0) -> None:
        self.tie_threshold = tie_threshold
    @abc.abstractmethod
    def judge(self, prompts, wtmk_images, candidate_images):
        pass

class CLIPQualityOracle(QualityOracle):
    def __init__(self, quality_model_name_or_path="adams-story/HPSv2-hf") -> None:
        super().__init__()
        self.quality_oracle = CLIPModel.from_pretrained(quality_model_name_or_path).to("cuda:1")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # uses the same exact vanilla clip processor

    def judge(self, prompts, wtmk_images, candidate_images):
        with torch.no_grad():
            # process inputs
            wtmk_inputs = self.clip_processor(text=prompts, images=wtmk_images, return_tensors="pt", padding=True, truncation=True)
            candidate_inputs = self.clip_processor(text=prompts, images=candidate_images, return_tensors="pt", padding=True, truncation=True)
            # move to device
            wtmk_inputs = {k: v.to("cuda:1") for k, v in wtmk_inputs.items()}
            candidate_inputs = {k: v.to("cuda:1") for k, v in candidate_inputs.items()}
            wtmk_outputs = self.quality_oracle(**wtmk_inputs)
            candidate_outputs = self.quality_oracle(**candidate_inputs)
            wtmk_logits_per_image = torch.diagonal(wtmk_outputs.logits_per_image) 
            candidate_logits_per_image = torch.diagonal(candidate_outputs.logits_per_image) 
        return wtmk_logits_per_image, candidate_logits_per_image

class ImageRewardQualityOracle(QualityOracle):
    def __init__(self, quality_model_name_or_path="ImageReward-v1.0", device="cuda:1") -> None:
        super().__init__()
        import ImageReward as reward
        self.device = device
        self.model = reward.load(quality_model_name_or_path).to(device)
        self.model.device = device
    
    def judge(self, prompts, wtmk_images, candidate_images):
        num_prompt = len(prompts)
        num_img = len(wtmk_images)
        _, wtmk_rewards = self.model.inference_rank(prompts, wtmk_images)
        _, candidate_rewards = self.model.inference_rank(prompts, candidate_images)
        wtmk_scores = torch.diagonal(torch.tensor(wtmk_rewards).reshape(num_prompt, num_img)).to(self.device)
        candidate_scores = torch.diagonal(torch.tensor(candidate_rewards).reshape(num_prompt, num_img)).to(self.device)
        return wtmk_scores, candidate_scores

class Trainer():
    def __init__(self, attack_steps=200, device='cuda', 
                 perturbation_model_name_or_path="runwayml/stable-diffusion-inpainting",
                 quality_model_name_or_path="adams-story/HPSv2-hf", choice_granularity=3) -> None:
        self.attack_steps = attack_steps

        self.perturbation_oracle = StableDiffusionInpaintPipeline.from_pretrained(
            perturbation_model_name_or_path,
            revision="fp16",
            torch_dtype=torch.float16,
            target_size=(1024, 1024),
            safety_checker=None,
            requires_safety_checker=False
        ).to("cuda:0")
        self.perturbation_oracle.scheduler = LMSDiscreteScheduler.from_config(self.perturbation_oracle.scheduler.config)
        self.quality_oracle = CLIPQualityOracle(quality_model_name_or_path)

    def attack(self, prompts, wtmk_images, num_images_per_prompt=1, batch_idx=1, guidance_scale=5, mask_ratio=0.02, 
                     num_inference_steps=75, save_interval=5, save_folder='results/'):
        images = copy.deepcopy(wtmk_images)
        os.makedirs(save_folder, exist_ok=True)
        valid_steps = 0
        batch_size = len(prompts)
        for attack_step in tqdm(range(self.attack_steps)):
            mask_images = [generate_square_mask(image, mask_ratio) for image in images]
            inpainted_images = self.perturbation_oracle(prompt=prompts, image=images, mask_image=mask_images, 
                                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                    num_images_per_prompt=num_images_per_prompt).images
            
            candidate_images = [interpolate_img(image, inpainted_image.resize((image.width,image.height)), mask_image=mask_image)
                                for image, inpainted_image, mask_image in zip(images, inpainted_images, mask_images)]
            wtmk_scores_per_image, candidate_scores_per_image = self.quality_oracle.judge(prompts, wtmk_images, candidate_images)
            better_image_mask = (wtmk_scores_per_image - candidate_scores_per_image) <= self.quality_oracle.tie_threshold 
            images = [candidate_images[i] if better_image_mask[i] else images[i] for i in range(batch_size)]
            valid_steps += better_image_mask.sum().item()

            if better_image_mask.sum() == better_image_mask.size():
                print(f"found a better image at step {attack_step}")

            if (attack_step+1) % save_interval == 0:
                for i, image in enumerate(images):
                    save_path = os.path.join(save_folder, f"img{batch_idx*batch_size+i}-step{attack_step+1}.png")
                    image.resize((1024, 1024)).save(save_path)
                    print("Saved image at", save_path)
            
        return images, valid_steps/batch_size

def main(scheme='stable-signature', sample_size=200, batch_size=20, attack_steps=200, mask_ratio=0.02,
         model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0', save_folder='./tmp'):
    trainer = Trainer(attack_steps=attack_steps, perturbation_model_name_or_path="stabilityai/stable-diffusion-2-inpainting") 
    detector = Detector(scheme=scheme)

    wtmk_results, attack_results = [], []
    attack_cnt = 0

    dataset = WatermarkedImageDataset(image_folder=os.path.join('./imgs', scheme), scheme=scheme, prompt_folder='./prompts', sample_size=sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)

    for batch_idx, batch in enumerate(dataloader):
        images, prompts = batch['image'], batch['prompt']
        images = [Image.open(image) for image in images]
        for img_idx, image in enumerate(images):
            image.save(os.path.join(f"{save_folder}", f"{scheme}", f"img{batch_idx*batch_size+img_idx}-original.png"))
        attack_images, valid_steps = trainer.attack(prompts, images, num_inference_steps=100, mask_ratio=mask_ratio,
                                                    save_folder=os.path.join(save_folder, f'{scheme}'), batch_idx=batch_idx)
        for prompt, attack_image, image in zip(prompts, attack_images, images):
            attack_image.save(os.path.join(save_folder, scheme, f"{attack_cnt}-attack.png"))
            attack_res = detector.detect(attack_image)
            attack_res['prompt'] = prompt
            attack_results.append(attack_res)
            attack_cnt += 1
            print("Attack result")
            print(attack_res)
            
            wtmk_res = detector.detect(image)
            wtmk_res['prompt'] = prompt
            wtmk_results.append(wtmk_res)
            print("Watermark result")
            print(wtmk_res)

    with open(os.path.join(save_folder, scheme, "wtmk_results.jsonl"), "w") as f:
        for res in wtmk_results:
            f.write(json.dumps(res) + '\n')
    with open(os.path.join(save_folder, scheme, "attack_results.jsonl"), "w") as f:
        for res in attack_results:
            f.write(json.dumps(res) + '\n')

if __name__ == '__main__':
    fire.Fire(main)