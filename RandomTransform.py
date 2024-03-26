import functools
from functools import singledispatch
from PIL import Image, ImageEnhance
import numpy as np
import random
import torch

from torchvision import transforms


def random_transform(image):
    """
    随机变换图像
    """
    actions = ['rotate', 'mirror', 'brightness', 'saturation', 'contrast', 'nothing']
    action = random.choice(actions)

    if action == 'rotate':
        angle = random.choice([0, 90, 180, 270])
        return image.rotate(angle)

    elif action == 'mirror':
        if random.choice([True, False]):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image.transpose(Image.FLIP_TOP_BOTTOM)

    elif action == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        factor = 0.5 + random.random()
        return enhancer.enhance(factor)

    elif action == 'saturation':
        enhancer = ImageEnhance.Color(image)
        factor = 0.5 + random.random()
        return enhancer.enhance(factor)

    elif action == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        factor = 0.5 + random.random()
        return enhancer.enhance(factor)

    elif action == 'nothing':
        return image


def randomTransformer(func):
    """
    装饰ImageLoader，将其读取的图像随机变换
    """
    @functools.wraps(func)
    def inner(*args, **kwargs):
        loader = func(*args, **kwargs)
        for batch, labels in loader:
            transformed_images = []
            for img_tensor in batch:
                img = transforms.ToPILImage()(img_tensor.squeeze_(0))
                transformed_img = random_transform(img)
                transformed_img_tensor = transforms.ToTensor()(transformed_img).unsqueeze_(0)
                transformed_images.append(transformed_img_tensor)
            transformed_batch = torch.cat(transformed_images, dim=0)
            yield transformed_batch, labels

    return inner
