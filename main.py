import json
import os
from typing import Tuple, List

import cv2
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import MNIST
from tqdm import tqdm
import json
import os
import torch.nn.functional as F

from torchvision import models
import torch
from torchvision.transforms import transforms
from PIL import Image

from one_neroro import Neroro


def get_data_loaders(path_to_save: str, train_batch_size: int = 5) -> Tuple[DataLoader, DataLoader]:
    train_set = MNIST(root=path_to_save, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_set = MNIST(root=path_to_save, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    return DataLoader(train_set, batch_size=train_batch_size), DataLoader(test_set)


def test(model, test_loader: DataLoader) -> float:
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            outputs = model(x)
            s_outputs = F.softmax(outputs, 1).data
            predicted = torch.argmax(s_outputs, 1)
            total += len(y)
            correct += (predicted == y).sum().item()
        accuracy = (correct / total) * 100
        print('Test Accuracy of the model: {} %'.format(accuracy))
    return accuracy


def train(model, data_loader: DataLoader, epoch_count: int, lr: float):
    model.train()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    timer = tqdm(range(epoch_count), desc="it's training time!")
    loss_list = []
    for _ in timer:
        avg_loss = 0.0
        for x, y in data_loader:
            outputs = model(x)
            outputs_loss = loss(outputs, y)
            s_outputs = F.softmax(outputs, 1).data
            avg_loss += outputs_loss.item()
            optimizer.zero_grad()
            outputs_loss.backward()
            optimizer.step()
            torch.argmax(s_outputs.data, 1)
        avg_loss /= len(data_loader)
        timer.set_description("loss:  " + str(f'{avg_loss:.4f}'), refresh=True)
        loss_list.append(avg_loss)
    return loss_list


IMG_DIR = 'images'
LABELS_PATH = 'imagenet-labels.json'


def load_labels(filename: str):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet


def get_batch(img_dir: str) -> Tuple[torch.Tensor, List[str]]:
    img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
    images = [Image.open(path) for path in img_paths]
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    images = [transform(image) for image in images]
    batch = torch.zeros(size=(len(images), 3, 224, 224))
    for i in range(len(images)):
        batch[i] = images[i]
    return batch, img_paths


def predict(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    vgg16_model = models.vgg16(pretrained=True)
    vgg16_model.eval()
    with torch.no_grad():
        output = vgg16_model(batch).data
        _, index = torch.max(output, 1)
        percentage = (F.softmax(output, dim=1)[0] * 100)[index]
    return index, percentage


def main():
    images_to_predict, path_to_images = get_batch(IMG_DIR)
    class_idx, percentage = predict(images_to_predict)
    labels_dictionary = load_labels(LABELS_PATH)
    for i in range(len(class_idx)):
        print('image: ', path_to_images[i])
        print('predicted_class: ', labels_dictionary[str(int(class_idx[i]))])
        print('confidence ', percentage[i], '%')


if __name__ == '__main__':
    main()

    a, b = get_data_loaders('/home/alex/Prog/Pro/Chiselki/DB', 1)
    models = (Neroro(), Neroro(), Neroro(), Neroro(), Neroro())
    train(models[0], data_loader=a, epoch_count=1, lr=0.001)
    train(models[1], data_loader=a, epoch_count=2, lr=0.001)
    train(models[2], data_loader=a, epoch_count=2, lr=0.002)
    train(models[3], data_loader=a, epoch_count=3, lr=0.002)
    train(models[4], data_loader=a, epoch_count=4, lr=0.004)
    print("ep:1, lr=0.001 | ", test(models[0], b))
    print("ep:2, lr=0.001 | ", test(models[1], b))
    print("ep:2, lr=0.002 | ", test(models[2], b))
    print("ep:3, lr=0.002 | ", test(models[3], b))
    print("ep:4, lr=0.004 | ", test(models[4], b))
