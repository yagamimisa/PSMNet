import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
 
    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    all_left_img = [each for each in sorted(os.listdir(filepath)) if each.endswith('LEFT_RGB.tif')]
    all_right_img =  [each for each in sorted(os.listdir(filepath)) if each.endswith('RIGHT_RGB.tif')]
    all_left_disp =  [each for each in sorted(os.listdir(filepath)) if each.endswith('LEFT_DSP.tif')]


    all_left_img = [filepath + '/' + x for x in all_left_img]
    all_right_img = [filepath + '/' + x for x in all_right_img]
    all_left_disp = [filepath + '/' + x for x in all_left_disp]
    
    return all_left_img, all_right_img, all_left_disp


