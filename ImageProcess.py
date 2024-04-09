import functools

import numpy as np
from PIL import Image, ImageEnhance
import random
import torch
from torchvision import transforms
import cv2


def image2tensor(image):
    """
    将图像转化为tensor
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze_(0)
    return image_tensor


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


def random_transform_tensor(images: torch.Tensor):
    """
    随机变换图像
    """
    transformed_images = []
    for img_tensor in images:
        img = transforms.ToPILImage()(img_tensor.squeeze_(0))
        transformed_img = random_transform(img)
        transformed_img_tensor = transforms.ToTensor()(transformed_img).unsqueeze_(0)
        transformed_images.append(transformed_img_tensor)
    transformed_batch = torch.cat(transformed_images, dim=0)
    return transformed_batch


def face_extract(img, definition=100, size=256):
    """
    :param img: 输入图像
    :param definition: 清晰度阈值
    :param size: 输出图像大小
    :return: [(face_resized, (x, y, w, h)), ...]
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    faces_with_details = []
    for (x, y, w, h) in faces:
        if w < 20 or h < 20:
            continue

        face = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (size, size), interpolation=cv2.INTER_NEAREST)

        # 检查清晰度（可定义清晰度）
        if np.mean(face_resized) > definition:  # 阈值
            faces_with_details.append((face_resized, (x, y, w, h)))

    return faces_with_details
