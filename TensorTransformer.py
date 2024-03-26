from functools import singledispatch
import torch

from torchvision import transforms


@singledispatch
def tensorTransformer(images: list):
    image_tensors = []
    for image in images:
        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensors.append(image_tensor)
    return torch.cat(image_tensors, dim=0)


@tensorTransformer.register(dict)
def _(images: dict):
    tensor_dict = {}
    for key in images.keys():
        image_tensors = []
        for image in images[key]:
            image_tensor = transforms.ToTensor()(image)
            image_tensor = image_tensor.unsqueeze_(0)
            image_tensors.append(image_tensor)
        tensor_dict[key] = torch.cat(image_tensors,dim=0)
    return tensor_dict
