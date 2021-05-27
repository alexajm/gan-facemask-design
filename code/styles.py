import torch
from torchvision.transforms import functional


def greyscale(img):
    img = functional.rgb_to_grayscale(img, num_output_channels=3) 
    return img


def blur(img):
    img = functional.gaussian_blur(img, kernel_size=3)
    return img
