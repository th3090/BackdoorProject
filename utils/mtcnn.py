import os

from facenet_pytorch import MTCNN
import torch
from facenet_pytorch.models.utils import training
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

data_dir = './dataset/test/'
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
    for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')


# Remove mtc

def main():
    print("Hello World!")

if __name__ == '__main__':
    main()
