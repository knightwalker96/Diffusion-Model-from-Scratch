# DIFFUSION MODEL FROM SCRATCH

This project implements a Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch on the MNIST dataset.

## Project Overview

Denoising Diffusion Probabilistic Models are a class of generative models that learn to produce data by reversing a gradual noising process. This implementation builds the core components of a DDPM, including:

* **Linear Noise Scheduler:** Manages the forward noising process (adding noise to images) and provides parameters for the reverse denoising process.
* **U-Net Model:** A U-Net architecture is used as the noise predictor. It takes a noisy image and a timestep as input and predicts the noise present in the image.
* **Training Script:** Trains the U-Net model to predict the added noise.
* **Sampling Script:** Generates new images by starting with random noise and iteratively applying the trained model to denoise it over a sequence of timesteps.

## File Structure

```text
.
├── data/                    # MNIST dataset will be downloaded here
│   └── MNIST/
│       └── raw/
├── default/                 # Directory for saving checkpoints and samples
│   ├── ddpm_ckpt.pth        
│   └── samples/             
├── noise_scheduler.py       # Implementation of the LinearNoiseScheduler
├── unet.py                  # Implementation of the U-Net model
├── training.py              # Script for training the DDPM
├── sampling.py              # Script for generating images
├── README.md                
└── requirements.txt         # (Recommended: Add a requirements file)
```
# USAGE
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/knightwaker96/Diffusion-Model-from-Scratch.git```
* ```cd Diffusion-Model-from-Scratch```
* ```pip install -r requirements.txt```
* For training/sampling use the below commands. 
* ```python training.py``` or ```python training.py --resume``` (if you want to if you want to resume training from the latest checkpoint)
* ```python sampling.py``` for generating images

#### Some details regarding the sampling process:
* The script loads the trained U-Net model from the default/ddpm_ckpt.pth checkpoint.
* It initializes the noise scheduler with the same parameters used during training.
* It will generate a batch of 100 images by performing the reverse diffusion process for 1000 steps. More the number of steps, better would be the sample quality but the tradeoff is time.
* The image x0_0.png will contain the final, most denoised generated samples.
