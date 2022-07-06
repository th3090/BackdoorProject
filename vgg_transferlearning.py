import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from models.vgg import *

# Device 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터 경로
train_dir = 'C:/Users/KIM/Desktop/YouTubeFaces/100training'
test_dir = 'C:/Users/KIM/Desktop/YouTubeFaces/100test'

# Hyperparameter
num_class = 100
IMG_SIZE = (224, 224)
workers = 0
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Transforms
transforms = transforms.Compose({
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
})

"""
Data_loader
"""

train_data = datasets.ImageFolder(train_dir, transform=transforms)
test_data = datasets.ImageFolder(test_dir, transform=transforms)

train_loader = DataLoader(
    train_data,
    num_workers=workers,
    batch_size=batch_size,
    shuffle=True)

test_loader = DataLoader(
    test_data,
    num_workers=workers,
    batch_size=batch_size)

"""
initial model
"""

model = vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=num_class, bias=True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=learning_rate)

"""
train
"""

for epoch in range(num_epochs):
    start = time.perf_counter()
    model.train()
    running_loss = 0.0
    correct_pred = 0
    for index, data in enumerate(train_loader):
        image, label = data
        image = image.to(device)
        label = label.to(device)
        y_pred = model(image)

        _, pred = torch.max(y_pred, 1)
        correct_pred += (pred == label).sum()

        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
    end = time.perf_counter()
    print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%'.
          format(epoch + 1, num_epochs, running_loss / (index + 1),
                 correct_pred.item() / (batch_size * (index + 1)) * 100))
    print('Time: {:.2f}s'.format(end - start))
print('Finished training!')

"""
test
"""
test_loss = 0.0
correct_pred = 0
for _, data in enumerate(test_loader):
    image, label = data
    image = image.to(device)
    lable = label.to(device)
    y_pred = model(image)

    _, pred = torch.max(y_pred, 1)
    correct_pred += (pred == label).sum()

    loss = criterion(y_pred, label)
    test_loss += float(loss.item())
print('Test loss: {:.4f}\tTest accuracy: {:.2f}%'.format(test_loss / 200, correct_pred.item() / 2000 * 100))
