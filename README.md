Stable Diffusion Implementation in PyTorch
========

This repository implements Stable Diffusion.
As of today the repo provides code to do the following:
* Training and Inference on Unconditional Latent Diffusion Models
* Training a Class Conditional Latent Diffusion Model
* Training a Text Conditioned Latent Diffusion Model
* Training a Semantic Mask Conditioned Latent Diffusion Model
* Any Combination of the above three conditioning

For autoencoder I provide code for vae as well as vqvae.
But both the stages of training use VQVAE only. One can easily change that to vae if needed

For diffusion part, as of now it only implements DDPM with linear schedule.


## Stable Diffusion Tutorial Videos
<a href="https://www.youtube.com/watch?v=1BkzNb3ejK4">
   <img alt="Stable Diffusion Tutorial" src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/7a24d114-38bd-43a8-9819-3afa112f39ab"
   width="400">
</a>
<a href="https://www.youtube.com/watch?v=hEJjg7VUA8g">
   <img alt="Stable Diffusion Conditioning Tutorial" src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/0b03cae4-4009-4bd7-8b02-328b0c5f33a3"
   width="400">
</a>
___  


## Sample Output for Autoencoder on CelebHQ
Image - Top, Reconstructions - Below

<img src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/2260d618-046e-411c-bea5-0c4cb7438560" width="300">

## Sample Output for Unconditional LDM on CelebHQ (not fully converged)

<img src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/212cd84a-9bd1-43f0-93b4-3b8ff9866571" width="300">

## Sample Output for Conditional LDM
### Sample Output for Class Conditioned on MNIST
![50](https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/46a38d36-3770-4f40-895a-95a16dc6462a)
![50](https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/1562c41d-e6ff-41cf-8d1e-6909a4240a04)
![50](https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/0cde44a6-746b-4f05-9422-9604f9436d91)
![50](https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/7d6b8db2-dab4-4a17-9fe6-570d938669f6)
![50](https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/6ecc3c61-3668-4305-aa4a-0f0e3cf815a0)

### Sample Output for Text(using CLIP) and Mask Conditioned on CelebHQ (not converged)
<img src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/78aa92bb-655e-46f6-92e9-a0c59787d700" width="100">
</br>
<img src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/9fcf66fb-65d7-4c2c-9fc4-9bbe428de11f" width="100">
Text -  She is a woman with blond hair
</br>
<img src="https://github.com/explainingai-code/StableDiffusion-PyTorch/assets/144267687/44ad67c8-c1e1-4ade-936f-18da92202e55" width="100">
Text -  She is a woman with black hair

___

## Setup
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/StableDiffusion-PyTorch.git```
* ```cd StableDiffusion-PyTorch```
* ```pip install -r requirements.txt```
* Download lpips weights from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and put it in ```models/weights/v0.1/vgg.pth```

___  

## Data Preparation
### Mnist

For setting up the mnist dataset follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Ensure directory structure is following
```
StableDiffusion-PyTorch
    -> data
        -> mnist
            -> train
                -> images
                    -> *.png
            -> test
                -> images
                    -> *.png
```

### CelebHQ 
#### Unconditional
For setting up on CelebHQ for unconditional, simply download the images from the official repo of CelebMASK HQ [here](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg

```
#### Mask Conditional
For CelebHQ for mask conditional LDM additionally do the following:

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg
            -> CelebAMask-HQ-mask-anno
                -> 0/1/2/3.../14
                    -> *.png
            
```

* Run `python -m utils.create_celeb_mask` from repo root to create the mask images from mask annotations

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg
            -> CelebAMask-HQ-mask-anno
                -> 0/1/2/3.../14
                    -> *.png
            -> CelebAMask-HQ-mask
                  -> *.png
```

#### Text Conditional
For CelebHQ for text conditional LDM additionally do the following:
* The repo uses captions collected as part of this repo - https://github.com/IIGROUP/MM-CelebA-HQ-Dataset?tab=readme-ov-file 
* Download the captions from the `text` link provided in the repo - https://github.com/IIGROUP/MM-CelebA-HQ-Dataset?tab=readme-ov-file#overview
* This will download a `celeba-captions` folder, simply move this inside the `data/CelebAMask-HQ` folder as that is where the dataset class expects it to be.

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg
            -> CelebAMask-HQ-mask-anno
                -> 0/1/2/3.../14
                    -> *.png
            -> CelebAMask-HQ-mask
                -> *.png
            -> celeba-caption
                -> *.txt
```
---
## Configuration
 Allows you to play with different components of ddpm and autoencoder training
* ```config/mnist.yaml``` - Small autoencoder and ldm can even be trained on CPU
* ```config/celebhq.yaml``` - Configuration used for celebhq dataset

Relevant configuration parameters

Most parameters are self explanatory but below I mention couple which are specific to this repo.
* ```autoencoder_acc_steps``` : For accumulating gradients if image size is too large for larger batch sizes
* ```save_latents``` : Enable this to save the latents , during inference of autoencoder. That way ddpm training will be faster

___  
## Training
The repo provides training and inference for Mnist(Unconditional and Class Conditional) and CelebHQ (Unconditional, Text and/or Mask Conditional).

For working on your own dataset:
* Create your own config and have the path in config point to images (look at `celebhq.yaml` for guidance)
* Create your own dataset class which will just collect all the filenames and return the image in its getitem method. Look at `mnist_dataset.py` or `celeb_dataset.py` for guidance 

Once the config and dataset is setup:
* Train the auto encoder on your dataset using [this section](#training-autoencoder-for-ldm)
* For training Unconditional LDM follow [this section](#training-unconditional-ldm)
* For class conditional ldm go through [this section](#training-class-conditional-ldm)
* For text conditional ldm go through [this section](#training-text-conditional-ldm)
* For text and mask conditional ldm go through [this section](#training-text-and-mask-conditional-ldm)


## Training AutoEncoder for LDM
* For training autoencoder on mnist,ensure the right path is mentioned in `mnist.yaml`
* For training autoencoder on celebhq,ensure the right path is mentioned in `celebhq.yaml`
* For training autoencoder on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py without conditining parts
* Map the dataset name to the right class in the training code [here](https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_ddpm_vqvae.py#L40)
* For training autoencoder run ```python -m tools.train_vqvae --config config/mnist.yaml``` for training vqvae with the desire config file
* For inference using trained autoencoder run```python -m tools.infer_vqvae --config config/mnist.yaml``` for generating reconstructions with right config file. Use save_latent in config to save the latent files 


## Training Unconditional LDM
Train the autoencoder first and setup dataset accordingly.

For training unconditional LDM map the dataset to the right class in `train_ddpm_vqvae.py`
* ```python -m tools.train_ddpm_vqvae --config config/mnist.yaml``` for training unconditional ddpm using right config
* ```python -m tools.sample_ddpm_vqvae --config config/mnist.yaml``` for generating images using trained ddpm

## Training Conditional LDM
For training conditional models we need two changes:
* Dataset classes must provide the additional conditional inputs(see below)
* Config must be changed with additional conditioning config added

Specifically the dataset `getitem` will return the following:
* `image_tensor` for unconditional training
* tuple of `(image_tensor,  cond_input )` for conditional training where cond_input is a dictionary consisting of keys ```{class/text/image}```

### Training Class Conditional LDM
The repo provides class conditional latent diffusion model training code for mnist dataset, so one
can use that to follow the same for their own dataset

* Use `mnist_class_cond.yaml` config file as a guide to create your class conditional config file.
Specifically following new keys need to be modified according to your dataset within `ldm_params`.
* ```  
  condition_config:
    condition_types: ['class']
    class_condition_config :
      num_classes : <number of classes: 10 for mnist>
      cond_drop_prob : <probability of dropping class labels>
  ```
* Create a dataset class similar to mnist where the getitem method now returns a tuple of image_tensor and dictionary of conditional_inputs.
* For class, conditional input will ONLY be the integer class
* ```
    (image_tensor, {
                    'class' : {0/1/.../num_classes}
                    })

For training class conditional LDM map the dataset to the right class in `train_ddpm_cond` and run the below commands using desired config
* ```python -m tools.train_ddpm_cond --config config/mnist_class_cond.yaml``` for training class conditional on mnist 
* ```python -m tools.sample_ddpm_class_cond --config config/mnist.yaml``` for generating images using class conditional trained ddpm

### Training Text Conditional LDM
The repo provides text conditional latent diffusion model training code for celebhq dataset, so one
can use that to follow the same for their own dataset

* Use `celebhq_text_cond.yaml` config file as a guide to create your config file.
Specifically following new keys need to be modified according to your dataset within `ldm_params`.
* ```  
    condition_config:
        condition_types: [ 'text' ]
        text_condition_config:
            text_embed_model: 'clip' or 'bert'
            text_embed_dim: 512 or 768
            cond_drop_prob: 0.1
  ```
* Create a dataset class similar to celebhq where the getitem method now returns a tuple of image_tensor and dictionary of conditional_inputs.
* For text, conditional input will ONLY be the caption
* ```
    (image_tensor, {
                    'text' : 'a sample caption for image_tensor'
                    })

For training text conditional LDM map the dataset to the right class in `train_ddpm_cond` and run the below commands using desired config
* ```python -m tools.train_ddpm_cond --config config/celebhq_text_cond.yaml``` for training text conditioned ldm on celebhq 
* ```python -m tools.sample_ddpm_text_cond --config config/celebhq_text_cond.yaml``` for generating images using text conditional trained ddpm

### Training Text and Mask Conditional LDM
The repo provides text and mask conditional latent diffusion model training code for celebhq dataset, so one
can use that to follow the same for their own dataset and can even use that train a mask only conditional ldm

* Use `celebhq_text_image_cond.yaml` config file as a guide to create your config file.
Specifically following new keys need to be modified according to your dataset within `ldm_params`.
* ```  
    condition_config:
        condition_types: [ 'text', 'image' ]
        text_condition_config:
            text_embed_model: 'clip' or 'bert
            text_embed_dim: 512 or 768
            cond_drop_prob: 0.1
        image_condition_config:
           image_condition_input_channels: 18
           image_condition_output_channels: 3
           image_condition_h : 512 
           image_condition_w : 512
           cond_drop_prob: 0.1
  ```
* Create a dataset class similar to celebhq where the getitem method now returns a tuple of image_tensor and dictionary of conditional_inputs.
* For text and mask, conditional input will be caption and mask image
* ```
    (image_tensor, {
                    'text' : 'a sample caption for image_tensor',
                    'image' : NUM_CLASSES x MASK_H x MASK_W
                    })

For training text unconditional LDM map the dataset to the right class in `train_ddpm_cond` and run the below commands using desired config
* ```python -m tools.train_ddpm_cond --config config/celebhq_text_image_cond.yaml``` for training text and mask conditioned ldm on celebhq 
* ```python -m tools.sample_ddpm_text_image_cond --config config/celebhq_text_image_cond.yaml``` for generating images using text and mask conditional trained ddpm


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of autoencoder the following output will be saved 
* Latest Autoencoder and discriminator checkpoint in ```task_name``` directory
* Sample reconstructions in ```task_name/vqvae_autoencoder_samples```

During inference of autoencoder the following output will be saved
* Reconstructions for random images in  ```task_name```
* Latents will be save in ```task_name/vqvae_latent_dir_name``` if mentioned in config

During training and inference of ddpm following output will be saved
* During training of unconditional or conditional DDPM we will save the latest checkpoint in ```task_name``` directory
* During sampling, unconditional sampled image grid for all timesteps in ```task_name/samples/*.png``` . The final decoded generated image will be `x0_0.png`. Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0
* During sampling, class conditionally sampled image grid for all timesteps in ```task_name/cond_class_samples/*.png``` . The final decoded generated image will be `x0_0.png`.  Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0
* During sampling, text only conditionally sampled image grid for all timesteps in ```task_name/cond_text_samples/*.png``` . The final decoded generated image will be `x0_0.png` . Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0
* During sampling, image only conditionally sampled image grid for all timesteps in ```task_name/cond_text_image_samples/*.png``` . The final decoded generated image will be `x0_0.png`. Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0




