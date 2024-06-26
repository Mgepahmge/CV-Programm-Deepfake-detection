import functools
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

__all__ = ['image2tensor', 'random_transform', 'random_transform_tensor', 'face_extract', 'getPowerSpectrum', 'getPowerSpectrumTensor']


def image2tensor(image):
    """
    将图像转化为tensor
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    image_tensor = transforms.ToTensor()(image_pil)
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


def azimuthalAverage(image, center=None):
    """
    计算图像的径向平均值


    :param image : 输入图像
    :param center : 中心坐标，默认为图像中心

    :return radial_prof (ndarray): 径向平均值分布
    """
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    r_int = r_sorted.astype(int)

    deltar = r_int[1:] - r_int[:-1]
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]

    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def getPowerSpectrum(img):
    """
    计算图像的功率谱
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    epsilon = 1e-8
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthalAverage(magnitude_spectrum)
    return torch.tensor(psd1D, dtype=torch.float)


def getPowerSpectrumTensor(img_tensor: torch.Tensor):
    """
    计算图像的功率谱
    """
    image_pil = transforms.ToPILImage()(img_tensor.squeeze_(0))
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    return getPowerSpectrum(image_cv)


if __name__ == '__main__':
    img = cv2.imread("00189.jpg")
    result = getPowerSpectrum(img)
    print(result.shape)
