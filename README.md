Stable Diffusion Implementation in PyTorch
========

This repository implements Stable Diffusion.
As of now this only implements unconditional latent diffusion models and trains on mnist and celebhq dataset.
Pretty soon it will also have code for conditional ldm.

For autoencoder I provide code for vae as well as vqvae.
But both the stages of training use VQVAE only. One can easily change that to vae if needed

For diffusion part, as of now it only implements DDPM with linear schedule.


## Stable Diffusion Videos



## Sample Output for Autoencoder on CelebHQ
Image - Top, Reconstructions - Below


## Sample Output for LDM on CelebHQ


## Data preparation
For setting up the mnist dataset:

Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

For setting up on CelebHQ, simply download the images from the official site.
And mention the right path in the configuration.


For training on your own dataset 
* Create your own config and have the path point to images (look at celebhq.yaml for guidance)
* Create your own dataset class, similar to celeb_dataset.py 
* Map the dataset name to the right class in the training code 


# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/StableDiffusion-PyTorch.git```
* ```cd StableDiffusion-PyTorch```
* ```pip install -r requirements.txt```
* Download lpips from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and put it in ```models/weights/v0.1/vgg.pth```
* For training autoencoder
* ```python -m tools.train_vqvae --config config/mnist.yaml``` for training vqvae
* ```python -m tools.infer_vqvae --config config/mnist.yaml``` for generating reconstructions
* For training ldm
* ```python -m tools.train_ddpm_vqvae --config config/mnist.yaml``` for training ddpm
* ```python -m tools.sample_ddpm_vqvae --config config/mnist.yaml``` for generating images

## Configuration
 Allows you to play with different components of ddpm and autoencoder training
* ```config/mnist.yaml``` - Small autoencoder and ldm can even be trained on CPU
* ```config/celebhq.yaml``` - Configuration used for celebhq dataset

Relevant configuration parameters

Most parameters are self explanatory but below I mention couple which are specific to this repo.
* ```autoencoder_acc_steps``` : For accumulating gradients if image size is too large for larger batch sizes
* ```save_latents``` : Enable this to save the latents , during inference of autoencoder. That way ddpm training will be faster

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of autoencoder the following output will be saved 
* Latest Autoencoder and discriminator checkpoint in ```task_name``` directory
* Sample reconstructions in ```task_name/vqvae_autoencoder_samples```

During inference of autoencoder the following output will be saved
* Reconstructions for random images in  ```task_name```
* Latents will be save in ```task_name/vqvae_latent_dir_name``` if mentioned in config

During training of DDPM we will save the latest checkpoint in ```task_name``` directory
During sampling, sampled image grid for all timesteps in ```task_name/samples/*.png``` 





