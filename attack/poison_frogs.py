from ast import Not
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import torchvision
from torchvision import datasets, models, transforms

import numpy as np

import cv2
from torch.autograd import Variable

from models.utils import training
from models.vgg_face import VGG_16

""" VGG 모델의 feature extractor 추출 함수 """


def vgg_feature_extractor(model, pretrained_path, output_classes=None):
    feature_extractor = model
    if output_classes is not None:
        feature_extractor.fc8 = nn.Linear(4096, output_classes)
    feature_extractor.load_state_dict(torch.load(pretrained_path))
    # feature_extractor = model
    feature_extractor.fc8 = nn.Identity()

    for name, module in feature_extractor.named_modules():
        for param in module.parameters():
            param.requires_grad = True

    #     for name, param in feature_extractor.named_parameters():
    #         print(name, param.requires_grad)

    return feature_extractor


""" Feature collision 수행을 위한 함수 """


def poison(feature_extractor, x, base_instance, target_instance, beta_0=0.2, lr=0.0001):
    """
    attack_instance x
    base_instance b
    target_instance t
    """

    # x = x.to(device)
    x.requires_grad = True

    feature_extractor.eval()

    fs_t = feature_extractor(target_instance.view(1, *target_instance.shape)).detach()
    fs_t.requires_grad = False

    beta = beta_0 * 4096 ** 2 / (3 * 224 * 224) ** 2  # 7.4

    # Forward Step:
    dif = feature_extractor(x.view(1, *x.shape)) - fs_t
    loss = torch.sum(torch.mul(dif, dif))
    loss.backward()

    x2 = x.clone()
    # x2-=(x.grad*lr)
    x2 -= (x.grad * lr)

    # Backward Step:
    x = (x2 + lr * beta * base_instance) / (1 + lr * beta)

    return x, loss.item()


