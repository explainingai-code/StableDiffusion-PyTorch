import glob
import os
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.latent_maps = None
        self.use_latents = False
        self.images = self.load_images(im_path)
        
        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, '*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, '*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, '*.{}'.format('jpeg')))
        for fname in fnames:
            ims.append(fname)
        print('Found {} images'.format(len(ims)))
        return ims
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            return latent
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()
        
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            return im_tensor