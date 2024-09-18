import torch
import argparse
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import numpy as np
import os
import torch.nn.functional as F

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Parsing Sampling Steps")
    parser.add_argument("steps", type=int)
    args = parser.parse_args()

    seed_value = 42
    set_seed(seed_value)

    bs = 128 # batch size
    tot = 50000 # total number of images

    model_path = "path/to/your/output/directory"
    unet = UNet2DConditionModel.from_pretrained(model_path + f"/checkpoint-{args.steps}/unet_ema", torch_dtype=torch.float16)
    unet.config.sample_size=32

    text_encoder = CLIPTextModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
            )
    text_encoder.requires_grad_(False)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16, safety_checker=None, num_images_per_prompt=bs)
    pipe.to("cuda")

    print("Creating output directory...")
    output_dir = f"imgs/{args.steps}"
    os.makedirs(output_dir, exist_ok=True)

    print("Reading Labels for Imagenet...")
    label_list = []

    with open('imagenet-classes.txt', 'r', encoding='utf-8') as file:
        for line in file:
            label = line.strip()
            label_list.append(label)

    assert len(label_list) == 1000, f"Expected 1000 label classes for imagenet-1k, but got {len(label_list)}."

    print("Generating images...")
    prompt_list = [''] * bs
    for i in range(tot // bs):
        for j in range(bs):
            prompt_list[j] = label_list[(i*bs+j) % 1000]
        
        images = pipe(prompt=prompt_list, num_inference_steps=20).images
        
        for j, image in enumerate(images):
            save_path = os.path.join(output_dir, f"{prompt_list[j]}_{(i*bs+j) // 1000}.png")
            image.save(save_path)
        print(f"Saved {(i+1)*bs} images!")

if __name__ == "__main__":
    main()