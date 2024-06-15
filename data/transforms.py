from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
import cv2


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                contrast_factor = np.random.uniform(0.8, 1.2)

            sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left'] = F.adjust_gamma(sample['left'], gamma)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['right'] = F.adjust_gamma(sample['right'], gamma)

        return sample


class RandomBrightness(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['left'] = F.adjust_brightness(sample['left'], brightness)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                brightness = np.random.uniform(0.5, 2.0)

            sample['right'] = F.adjust_brightness(sample['right'], brightness)

        return sample


class RandomHue(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left'] = F.adjust_hue(sample['left'], hue)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                hue = np.random.uniform(-0.1, 0.1)

            sample['right'] = F.adjust_hue(sample['right'], hue)

        return sample


class RandomSaturation(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_saturation(sample['left'], saturation)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                saturation = np.random.uniform(0.8, 1.2)

            sample['right'] = F.adjust_saturation(sample['right'], saturation)

        return sample


class RandomColor(object):

    def __init__(self,
                 asymmetric_color_aug=False,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, img_left, img_right):
        transforms = [RandomContrast(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomGamma(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomBrightness(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomHue(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomSaturation(asymmetric_color_aug=self.asymmetric_color_aug)]

        sample = {"left": img_left, "right": img_right}
    
        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        return sample["left"], sample["right"]
