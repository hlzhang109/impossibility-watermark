from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
import torch
from tqdm import tqdm
import math
from detector import Detector
import fire
import torch.distributed as dist

class Generator():
    def __init__(self, model_name_or_path="stabilityai/sdxl-turbo", scheme='stable-signature', wtmk_strength='strong', target_size=(1024, 1024), device='cuda'):
        self.scheme = scheme
        if scheme == 'stable-signature':
            vae = AutoencoderKL.from_pretrained(f"imatag/stable-signature-bzh-sdxl-vae-{wtmk_strength}")
            self.model = StableDiffusionXLPipeline.from_pretrained(model_name_or_path, vae=vae, height=1024, width=1024, target_size=target_size, device_map='auto')
        elif scheme == 'invisible-watermark':
            self.model = StableDiffusionXLPipeline.from_pretrained(model_name_or_path, add_watermarker=True, target_size=target_size, torch_dtype=torch.float16, device_map='auto')
            # Copied from https://github.com/huggingface/diffusers/blob/05be622b1c152bf11026f97f083bb1b989231aec/src/diffusers/pipelines/stable_diffusion_xl/watermark.py#L17
            self.WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
            # bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
            self.WATERMARK_BITS = [int(bit) for bit in bin(self.WATERMARK_MESSAGE)[2:]]
        else:
            raise NotImplementedError(f"Scheme {scheme} not implemented")
    
    def generate(self, prompt, guidance_scale=7.5, num_inference_steps=50):
        return self.model(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images
    
    def distributed_generate(self, rank, world_size, prompt, guidance_scale=7.5, num_inference_steps=50):
        # create default process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        # move to rank
        self.model.to(rank)

        torch.distributed.get_rank()

def generate_watermarked_samples(batch_size=20, sample_size=200, prompts_file_name="prompts/prompts.txt",
                                 scheme='stable-signature', model_name_or_path='stabilityai/sdxl-turbo', save_folder='./imgs'):
    generator = Generator(model_name_or_path, scheme=scheme)
    detector = Detector(scheme=scheme)
    with open(prompts_file_name, "r") as f:
        all_prompts = f.readlines()
    all_prompts = all_prompts[:sample_size]
    if scheme=='stable-signature':
        model_name_or_path='stabilityai/sdxl-turbo'
    else:
        model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0'
    if model_name_or_path=='stabilityai/sdxl-turbo':
        num_inference_steps, guidance_scale = 4, 0.0
    else:
        num_inference_steps, guidance_scale = 50, 7.5

    for batch in tqdm(range(math.ceil(sample_size/batch_size))):
        prompts = all_prompts[batch*batch_size:(batch+1)*batch_size]
        wtmk_images = generator.generate(prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) 
        detected_result = [detector.detect(wtmk_image) for wtmk_image in wtmk_images]
        for j, wtmk_image in enumerate(wtmk_images):
            wtmk_image.save(f"{save_folder}/{batch*batch_size+j}-original.png")
        print(f"Batch {batch} [{batch_size*batch}/{sample_size}] finished.")
        print(detected_result)

if __name__ == "__main__":
    fire.Fire(generate_watermarked_samples)