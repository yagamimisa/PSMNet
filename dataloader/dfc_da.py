import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import preprocess 
import listflowfile as lt
import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')
def grey_loader(path):
    return Image.open(path).convert('L')
class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader,loader_g=grey_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.training = training
        self.loader_g = loader_g
    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]
	left_img = self.loader(left)
        right_img = self.loader(right)
        disp_img = self.loader_g(disp_L)
	processed = preprocess.get_transform(augment=False)  

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
	   disp_img = np.ascontiguousarray(disp_img,dtype=np.float32)
           disp_img = disp_img[y1:y1 + th, x1:x1 + tw]


           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, disp_img

        else:
	   w, h = left_img.size

           left_img = left_img.crop((0, 0, 1024,1024))
           right_img = right_img.crop((0,0,1024,1024))
           disp_img = np.ascontiguousarray(disp_img,dtype=np.float32)
          


           processed = preprocess.get_transform(augment=False)
           left_img   = processed(left_img)
           right_img  = processed(right_img)
	   disp_img = disp_img[0:1024,0:1024]
           return left_img, right_img, disp_img

		
 
    def __len__(self):
        return len(self.left)
