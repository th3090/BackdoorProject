import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
import cv2


def img_read(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def image_compare_show(base_image, target_image):
    """
    Tensor로 변환 전 기존 이미지 비교를 위한 시각화 함수
    :param base_image: target_image의 특성을 받을 이미지
    :param target_image: base_image에 특성을 전달할 이미지
    :return: base_image와 target_image의 초기 상태 시각화
    """
    fig = plt.figure(figsize=(15, 8))
    rows = 1
    cols = 4

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(base_image)
    ax1.set_title('base_image')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(target_image)
    ax2.set_title('target_image')

    plt.subplots_adjust(wspace=0.5)

    plt.show()


def img_to_tensor(image, img_size):
    """
    이미지를 tensor로 변환하고 image size를 조정하는 함수
    :param image: 변환할 이미지
    :param img_size: Model에 입력하기 위해 이미지 사이즈 조정
    :return: 원본 이미지를 tensor 형태로 변환하여 반환

    Example :
    base_instance = img_trans_to_tensor(base_instance, IMG_SIZE)
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size)
    ])
    image = trans(image)

    return image


def tensor_imshow(instance, title=None):
    """
    Tensor로 변환된 instance를 plt로 보기 위한 함수
    :param instance: Tensor로 변환된 instance -> gpu에 있을 시 .detach.cpu() 필요
    :param title: Image title

    Example :
    base_instance = base_instance.to(device)
    list = [attack_instance, base_instance, target_instance]
    out = torchvision.utils.make_grid(list)
    pf.imshow(out.detach().cpu(), title = "attack - base - target")
    """
    instance = instance.numpy().transpose((1, 2, 0))  # array 순서 변경
    instance = np.clip(instance, 0, 1)  # 값 고정
    plt.imshow(instance)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def attack_image_save(instance, path, name):
    """
    tensor 연산 후 생성된 이미지를 저장하는 함수
    :param instance: 이미지로 변환할 인스턴스
    :param path: 이미지를 저장할 경로
    :param name: 저장할 이미지의 이름
    """
    image = instance.detach().cpu()
    torchvision.utils.save_image(image, os.path.join(path, name))


def make_torch_grid(*args):
    instance_list = [*args]
    out = torchvision.utils.make_grid(instance_list)
    return out

# # Get a batch of training data
# inputs, classes = next(iter(train_loader))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
#
# imshow(out)
