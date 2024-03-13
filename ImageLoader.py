import os
from PIL import Image
import torch
from torchvision import transforms


def imageLoader(folder_path: str, label: int, step: int = 1):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            image_tensor = transforms.ToTensor()(image)
            image_tensor = image_tensor.unsqueeze_(0)
            images.append(image_tensor)
            labels.append(label)
            if len(images) == step:
                batch = torch.cat(images, dim=0)
                yield batch, labels
                images = []
                labels = []


if __name__ == "__main__":
    path = "D:/Python项目/CV-Programm-Deepfake-detection/dataset/real_vs_fake/real-vs-fake/test/fake"
    loader = imageLoader(path, 1, 10)
    files, labels = next(loader)
    print(f"{files.shape} and {labels}")
