<div align="center">

# Immiscible Diffusion: Stable Diffusion Version
*Yiheng Li, Heyang Jiang, Akio Kodaira, Masayoshi Tomizuka, Kurt Keutzer, Chenfeng Xu*

This folder contains the implementation of Immiscible Diffusion on Stable Diffusion [^1]. For general information regarding the codebase, please refer to the original repo [^1] we built on.
</div>

## Get Started

### Installing the dependencies: official way

Please refer to [^1] for the environment you need. There might be few packages that need to be installed manually. Here we quote a few important steps from [^1]:

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

### Installing the dependencies: our environment

For the user's convenience, we also extract our conda environment to `sd.yml`. Note this could contain unnecessary packages or packages not matching your system (CUDA version etc.). To use this, you can run the following scripts to create an environment named `sd`

```bash
conda env create -f sd.yml
```

## Train From Scratch

Use bash file `train_scratch.sh` to start the training program `conditional_scratch_train_sd.py`. Note that you need to carefully read all coefficients in the bash file to ensure the program runs smoothly on your device. Refer to [^1] for the explanation of coefficients. For this experiment, we use 8 * Nvidia A800 GPUs with a total batch size of 2,048 = 256 * 8.

## Fine-tuning

Use bash file `train_ft.sh` to start the fine-tuning program `conditional_ft_train_sd.py`. Note that you need to carefully read all coefficients in the bash file to ensure the program runs smoothly on your device. Refer to [^1] for the explanation of coefficients. For this experiment, we use 4 * Nvidia A6000 GPUs with a total batch size of 512 = 128 * 4.

## Sampling

Repo [^1] provides a simple sampling program for generating 1 image. We modify it as follows,

```python
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "path_to_saved_model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet", torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained("<initial model>", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="CLASS_NAME").images[0]
image.save("image.png")
```

We also provide a large-quantity sampling program for FID calculation purposes. Please use bash file `generate.sh` to start the sampling program `generate.py`. Note that you need to carefully read the bash and py file to ensure the program runs smoothly on your device. For this experiment, we use 1 * Nvidia A5000 GPU with a total batch size of 128.

## Performance

With class-conditional generation on Stable Diffusion, we observe FIDs as follows,

| Task | Batch Size | Training Steps | Vanilla SD FID | Immiscible SD FID |
|------|------------|----------------|----------------|-------------------|
| Training from Scratch | 2048 = 256 * 8 | 20k | 17.92 | 16.43 |
| Fine-tuning on Stable Diffusion v1.4 [^1] | 512 = 128 * 4 | 5k | 11.45 | 10.28 |

We observe diverse methods for computing FIDs. For consistency, we updated FIDs reported here to be evaluated with pytorch-fid[^2].

## Citation
If this work is helpful for your research, please consider citing:

```
@misc{li2024immisciblediffusionacceleratingdiffusion,
      title={Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment}, 
      author={Yiheng Li and Heyang Jiang and Akio Kodaira and Masayoshi Tomizuka and Kurt Keutzer and Chenfeng Xu},
      year={2024},
      eprint={2406.12303},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.12303}, 
}
```

## Acknowledgement

This implementation is built upon:

[^1]: Stable Diffusion: [diffusers/examples/text_to_image](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

[^2]: [Pytorch-FID](https://github.com/mseitzer/pytorch-fid)
