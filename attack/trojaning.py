import imageio.v2 as imageio
from utils.images import *
from models.vgg import *
import torch.nn as nn
import torch
import copy
import numpy as np


def filter_part(w, h):
    # masks = []

    # square trojan trigger shape
    mask = np.zeros((h, w))
    for y in range(0, h):
        for x in range(0, w):
            if w - 80 < x < w - 20 and h - 80 < y < h - 20:
                mask[y, x] = 1
    # masks.append(np.copy(mask))

    # # Apple logo trigger shape
    # data = imageio.imread('./dataset/apple4.pgm')
    # mask = np.zeros((h, w))
    # for y in range(0, h):
    #     for x in range(0, w):
    #         if w - 105 < x < w - 20 and h - 105 < y < h - 20:
    #             if data[y - (h - 105), x - (w - 105)] < 50:
    #                 mask[y, x] = 1
    # masks.append(np.copy(mask))
    #
    # # watermark trigger shape
    # data = imageio.imread('./dataset/watermark3.pgm')
    # mask = np.zeros((h, w))
    # for y in range(0, h):
    #     for x in range(0, w):
    #         if data[y, x] < 50:
    #             mask[y, x] = 1
    #
    # masks.append(np.copy(mask))
    return mask


def neurons_searching(base_model):
    """
    Target layer의 다음 layer에서 weights 값을 추출하여 numpy 형태로 변환 후 _get_largest_value_neuron
    에 weights 값을 전달하고, 최종적으로 최대 값을 갖는 뉴런의 인덱스를 반환
    :param base_model: 기존의 Pretrained model
    :return: Largest weights value neuron index
    """
    # Case 1. VGG16 모델에서 첫번째 FC layer를 target layer로 삼는 경우
    weights = base_model.classifier[3].weight.T
    weights = weights.detach().numpy()
    return _get_largest_value_neuron(weights)


def _get_largest_value_neuron(weights):
    """
    Weight 절대값 합산 후 최대 값을 갖는 뉴런의 인덱스를 반환하는 함수
    :param weights: neurons_searching 함수로 부터 target layer의 다음 layer의 가중치 값을 받아옴
    :return: 절대값 합산 후 최대 값을 갖는 뉴런의 인덱스를 반환
    """
    neurons_weight = np.sum(np.abs(weights), axis=1)
    largest_value_neuron = neurons_weight.argmax()
    return largest_value_neuron
    # print('neuron sort', np.argsort(neurons_weight)[-10:])


def target_layer_model(base_model):
    """
    Target layer (ex: fc6 , conv5)까지의 모델을 복사하는 함수
    :param base_model: [  : Target layer]를 추출할 모델
    :return: Target Layer까지의 layer를 가지는 새로운 모델
    TODO: Convolutional layer 확인
    """
    # Case 1. VGG16 모델에서 첫번째 FC layer를 target layer로 삼는 경우
    model = copy.deepcopy(base_model)
    model.classifier = nn.Sequential(*list(base_model.classifier.children())[:1])

    return model


def set_start_image(img_size):
    """
    Trojan image를 만들기 위해 초기 이미지 셋팅
    :param img_size: 이미지 사이즈
    :return: start_image (numpy array)를 반환
    """
    seed = 1
    np.random.seed(seed)
    background_color = np.float32([175.0, 175.0, 175.0])
    start_image = np.random.normal(background_color, 8, (img_size[0], img_size[1], 3))
    start_image = start_image/255.0
    return start_image


def save_image(output_folder, filename, neuron_number, img):
    """
    생성된 이미지를 저장하기 위한 함수
    :param output_folder: 생성된 이미지를 저장할 폴더
    :param filename: 파일명
    :param neuron_number: Target neuron number
    :param img: 저장할 이미지
    :return: 저장 경로를 리턴
    """
    path = "%s/%s_%s.jpg" % (output_folder, filename, str(neuron_number).zfill(4))
    imageio.imsave(path, img)

    return path


def set_vari(model, instance, target_neuron):
    """
    Target layer까지의 모델에서 neuron activation value를 추출하는 함수
    :param model: Target layer까지의 모델
    :param instance: 모델에 넣을 인스턴스
    :param target_neuron: 값을 바꿀 뉴런
    """
    layer_activation = model(instance.view(1, *instance.shape))[0]
    largest_neuron_index = layer_activation.argmax()
    largest_neuron_act_value = layer_activation[largest_neuron_index]
    target_neuron_act_value = layer_activation[target_neuron]

    # one_hot = np.zeros_like(layer_activation.detach().numpy())
    # one_hot[target_neuron] = 1.0

    return largest_neuron_index, largest_neuron_act_value, target_neuron_act_value


def make_mask(image, w, h, img_size):
    """
    Filter_part에서 위치를 지정한 mask(2d)에 (0 or 1) 값을 채우고, tensor로 변환하는 함수
    :param image: 그냥 이미지 타입을 위해서 넣어준 것임, 모델에 삽입하기 위한 이미지 크기
    :param w: 이것도 결국 224
    :param h: 이것도 결국 224
    :param img_size: 이것도 반복됨
    """
    mask = np.zeros_like(image)
    mask_filter = filter_part(w, h)

    for y in range(h):
        for x in range(w):
            if mask_filter[y][x] == 1:
                mask[x, y, :] = 1

    return img_to_tensor(mask, img_size)


def trojan_attack(model, instance, target_neuron, mask, target_value=100., lr=0.0001):
    """
    Trojan attack을 수행하기 위한 함수
    :param model: Base model에서 target layer까지만 추출한 모델
    :param instance: Start image로 부터 만들어 낸 initial instance
    :param target_neuron: Pretrained model의 target layer 바로 다음 layer에서 가장 큰 activation value를 갖는 neuron index
    :param mask: Image를 변환시킬 구역
    :param target_value: Loss function을 위한 임의 지정 값
    :param lr: Learning rate
    """
    instance.requires_grad = True

    model.eval()

    activation_vector = model(instance.view(1, *instance.shape))[0]
    target_neuron_value = activation_vector[target_neuron]

    dif = target_value - target_neuron_value
    loss = torch.sum(torch.mul(dif, dif))
    loss.backward()

    temp_instance = instance.clone()
    temp_instance -= instance.grad * mask * lr

    instance = temp_instance

    return instance, loss.item()


def trigger_extract(instance, mask):
    """
    Activation value 변환이 끝난 instance로 부터 trigger를 추출하기 위한 함수
    :param instance: trojan_attack 수행이 완료된 instance
    :param mask: Trigger를 추출할 위치 (masking)
    """
    instance = instance.detach().cpu()
    mask = mask.detach().cpu()
    trigger = instance * mask
    return trigger


def trigger_injection(base_image, mask, trigger, img_size, transparancy=0):
    """
    추출한 trigger를 image에 합성하는 함수
    :param base_image: Trigger를 삽입할 기본 이미지
    :param mask: Trigger를 주입할 위치 (masking)
    :param trigger: Trigger_extract 함수 반환 값
    :param img_size: 이미지 사이즈
    :param transparancy: 투명도 조절을 위한 변수
    """
    mask = mask.detach().cpu()
    attack_instance = img_to_tensor(base_image, img_size)
    if transparancy != 0:
        attack_instance = (1 - mask) * attack_instance + mask * trigger * transparancy\
                          + mask * attack_instance * (1 - transparancy)
    elif transparancy > 1:
        raise AssertionError('Transparancy value is between 0 and 1')
    else:
        attack_instance = (1 - mask) * attack_instance + mask * trigger

    return attack_instance


def main():
    img_size = (224, 224)
    path = '/dataset/test/'

    base = './dataset/base.jpg'
    base_image = img_read(base)
    # instance = img_to_tensor(image, img_size)
    # instance = instance.view(1, *instance.shape)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    base_model = vgg16(pretrained=True)
    model = target_layer_model(base_model)
    neuron_index = neurons_searching(base_model)

    start_image = set_start_image(img_size)
    start_instance = img_to_tensor(start_image, img_size)
    start_instance = start_instance.type(torch.FloatTensor)
    mask = make_mask(start_image, 224, 224, img_size)
    mask = mask.type(torch.FloatTensor)

    model = model.to(device)
    start_instance = start_instance.to(device)
    mask = mask.to(device)

    instance = start_instance

    for i in range(5000):
        instance, loss = trojan_attack(model, instance.detach(), neuron_index, mask)
        if i % 100 == 0:
            print(loss)

    attack_image_save(instance, path, "trojan_instance_iter5000.jpg")
    attack_image_save(start_instance, path, "start_instance.jpg")
    attack_image_save(mask, path, "mask.jpg")

    a, b, c = set_vari(model, instance, neuron_index)
    print("Start_instance의 largest activation neuron index : {}".format(a))
    print("Start_instance의 largest activation neuron value : {}".format(b))
    print("Start_instance의 target neuron index : {}".format(neuron_index))
    print("Start_instance의 target neuron value : {}".format(c))

    print("=================================================")
    d, e, f = set_vari(model, start_instance, neuron_index)
    print("Trojan_instance의 largest activation neuron index : {}".format(d))
    print("Trojan_instance의 largest activation neuron value : {}".format(e))
    print("Trojan_instance의 target neuron index : {}".format(neuron_index))
    print("Trojan_instance의 target neuron value : {}".format(f))

    trigger = trigger_extract(instance, mask)
    attack_instance = trigger_injection(base_image, mask, trigger, img_size, path)

    attack_image_save(attack_instance, path, "Trigger_injection.jpg")


if __name__ == '__main__':
    main()
