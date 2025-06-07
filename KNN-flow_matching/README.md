<div align="center">

# Immiscible Diffusion: KNN Implementation on Flow Matching

This folder contains the KNN implementation of Immiscible Diffusion on Flow Matching [^1]. For general information regarding the codebase, please refer to the original repo [^1] we built on.
</div>

## Get Started

### Installing the dependencies: official way

Please refer to [^1] for the environment you need. There might be few packages that need to be installed manually. Here we quote a few important steps from [^1]:

Before running the scripts, make sure to install the library's training dependencies:

```
# [OPTIONAL] create conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# install torchcfm
pip install -e .
```

### Installing the dependencies: our environment

For the user's convenience, we also extract our conda environment to `fm.yml`. Note this could contain unnecessary packages or packages not matching your system (CUDA version etc.). To use this, you can run the following scripts to create an environment named `fm`

```bash
conda env create -f fm.yml
```

## Train From Scratch

Use bash file `train_otcfm.sh` to start the training. Note that you need to carefully read all coefficients in the bash file to ensure the program runs smoothly on your device. Refer to [^1] for the explanation of coefficients. For this experiment, we use 1 * Nvidia A6000 GPUs with a total batch size of 256.

## Sampling and Evaluation

Please use bash file `fid.sh` or `fid_euler.sh` to start the sampling and evaluation with adaptive and euler solvers respectively. Note that you need to carefully read the bash and py file to ensure the program runs smoothly on your device. For this experiment, we use 1 * Nvidia A6000 GPU with a total batch size of 1,024.

## Performance

With flow matching, we observe FIDs as follows,

| Solver | Batch Size | Training Steps | Vanilla SD FID | Immiscible SD FID (KNN, k=4) |
|------|------------|----------------|----------------|-------------------|
| Euler 50 Steps | 256 | 20k | 5.07 | 3.68 |
| Adaptive Solver | 256 | 20k | 3.49 | 3.31 |

## Key Implementations

The KNN implementations are in `knnpackage/torchcfm/optimal_transport.py`, `line 144 - 163`, replacing the `sample_plan` function of `OTPlanSampler` class. The **k** value can be altered at `line 144` of `knnpackage/torchcfm/optimal_transport.py`.

## Acknowledgement

This implementation is built upon:

[^1]: TorchCFM: [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) [Image Generation Task](https://github.com/atong01/conditional-flow-matching/tree/main/examples/images/cifar10)
