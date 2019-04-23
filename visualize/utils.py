import torch

from PIL import Image
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt


def load_image(path):
    image = Image.open(path)
    image = F.to_tensor(image)
    return image.unsqueeze(0)


def plot_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


def normalize_image(tensor):
    min_tensor = torch.min(tensor)
    range_tensor = torch.max(tensor) - min_tensor
    return (tensor - min_tensor) / range_tensor
