import random
import torch
from torchvision.transforms import transforms

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

pattern_tensor: torch.Tensor = torch.tensor([
    [1., 0., 1., 1., 0., 1.],
    [-10., 1., -10., -10., 1., -10.],
    [-10., -10., 0., -10., -10., 0.],
    [-10., 1., -10., -10., 1., -10.],
    [1., 0., 1., 1., 0., 1.],
    [1., 0., 1., 1., 0., 1.],
    [-10., 1., -10., -10., 1., -10.],
    [-10., -10., 0., -10., -10., 0.],
    [-10., 1., -10., -10., 1., -10.],
    [1., 0., 1., 1., 0., 1.]
])


x_top = 3
y_top = 23

mask_value = -10
input_shape = [3, 224, 224]

# mask: torch.Tensor = None
# pattern: torch.Tensor = None


def make_pattern_image(pattern, x, y):
    image = torch.zeros(input_shape)
    image.fill_(mask_value)

    x_bot = x + pattern.shape[0]  # pattern_tensor.shape[0] : 가로 크기
    y_bot = y + pattern.shape[1]  # pattern_tensor.shape[1] : 세로 크기

    # self.params.input_shape[1 or 2] : input image의 x, y 한계 지점
    if x_bot >= input_shape[1] or y_bot >= input_shape[2]:
        raise ValueError(f'Position of backdoor outside image limits:'
                         f'image: {input_shape}, but backdoor'
                         f'ends at ({x_bot}, {y_bot})')

    # pattern 삽입
    image[:, x_top:x_bot, y_top:y_bot] = pattern

    return image


def injection_pattern(input_image, pattern_image):
    pattern_mask = 1 * (pattern_image != mask_value)
    backdoored_image = (1 - pattern_mask) * input_image + pattern_mask * pattern_image
    return backdoored_image

