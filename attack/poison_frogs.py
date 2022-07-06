import torch
import torch.nn as nn

from models.vgg import *
from models.resnet import *
from utils.images import img_read, img_to_tensor, make_torch_grid, tensor_imshow


def feature_extractor(arch, model):
    if arch == 'vggnet':
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

    elif arch == 'resnet':
        model = nn.Sequential(*list(model.children())[:-2])

    # assert not arch != 'vggnet' or 'resnet'

    # model.eval()

    return model


def poison_frogs(extractor, attack_instance, base_instance, target_instance, beta_0=0.2, lr=0.0001):

    # x = x.to(device)
    attack_instance.requires_grad = True  # base_instance로 부터 target_instance의 특성을 물려 받을 instance

    extractor.eval()

    # target_instance의 feature 추출 & attack_instance의 feature 추출
    target_instance_feature = extractor(target_instance.view(1, *target_instance.shape))[0].detach()  # target_instance의 feature를 추출하여 저장 (gradient 고정)
    target_instance_feature.requires_grad = False  # .detach() 사용할 경우 gradient 전파 안되는 텐서를 생성

    attack_instance_feature = extractor(attack_instance.view(1, *attack_instance.shape))[0]

    # Forward Step:
    dif = attack_instance_feature - target_instance_feature
    loss = torch.sum(torch.mul(dif, dif))
    loss.backward()

    forward_attack_instance = attack_instance.clone()
    forward_attack_instance -= (attack_instance.grad * lr)

    # 원본 이미지 유지를 위한 hyperparameter 설정
    beta = beta_0 * 4096 ** 2 / (3 * 224 * 224) ** 2  # 7.4

    # Backward Step:
    attack_instance = (forward_attack_instance + lr * beta * base_instance) / (1 + lr * beta)

    return attack_instance, loss.item()


def feature_loss_compare(extractor, instance_a, instance_b):
    instance_a_feature = extractor(instance_a.view(1, *instance_a.shape))[0]
    instance_b_feature = extractor(instance_b.view(1, *instance_b.shape))[0]

    dif = instance_a_feature - instance_b_feature
    loss = torch.sum(torch.mul(dif, dif))

    return loss.item()


#
# def poison_res(arch, feature_extractor, attack_instance, base_instance, target_instance, beta_0=0.2, lr=0.0001):
#
#     return


# x = base_instance
# for i in range(1000):
#     x, loss = poison(feature_space, x.detach(), base_instance, target_instance)
#     if i % 50 == 0:
#         print(loss)


def main():
    # cuda 사용을 위한 device 지정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # input image 지정
    img_size = (224, 224)

    # image 경로
    base = './dataset/base.jpg'
    target = './dataset/target.jpg'

    # image read (cv2 , BGR to RGB)
    base_image = img_read(base)
    target_image = img_read(target)

    # image를 tensor 형태로 변환
    base_instance = img_to_tensor(base_image, img_size)
    target_instance = img_to_tensor(target_image, img_size)

    # 사용할 모델 지정 및 pretrained extractor 추출
    model = vgg16(pretrained=True)
    # model = resnet50(pretrained=True)
    extractor = feature_extractor('vggnet', model)
    # extractor = feature_extractor('resnet', model)

    # GPU 사용 (cuda)를 위해 extractor 및 instance gpu로 이동
    extractor = extractor.to(device)
    base_instance = base_instance.to(device)
    target_instance = target_instance.to(device)

    # poison frogs 를 위해 초기 attack_instance (attack sample)을 base_instance로 초기화
    attack_instance = base_instance
    for i in range(200):
        attack_instance, loss = poison_frogs(extractor, attack_instance.detach(), base_instance, target_instance)
        if i % 50 == 0:
            print(loss)

    # attack, base, target instance 비교를 위한 시각화
    out = make_torch_grid(attack_instance, base_instance, target_instance)
    tensor_imshow(out.detach().cpu(), title="attack - base - target")

    # 각 instance 별 loss값 비교
    print("Base instance - Initial Attack instance: {}".format(feature_loss_compare(extractor, base_instance, base_instance)))
    print("Base instance - Target instance: {}".format(feature_loss_compare(extractor, base_instance, target_instance)))
    print("Base instance - Attack instance: {}".format(feature_loss_compare(extractor, base_instance, attack_instance)))
    print("Target instance - Attack instance: {}".format(feature_loss_compare(extractor, target_instance, attack_instance)))


if __name__ == '__main__':
    main()

