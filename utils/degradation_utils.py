import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np

from utils.image_utils import crop_img


class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        return clean_patch_1, clean_patch_2

    # Keep this for backward compatibility if any old code calls it
    def _degrade_by_type(self, clean_patch, degrade_type):
        return clean_patch, clean_patch
