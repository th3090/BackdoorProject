import os
import glob
from PIL import Image


#  이미지 파일 이름을 폴더명 + @로 변경
def image_name_change(data_dir):
    data_folder = os.listdir(data_dir)
    for folder in data_folder:
        files = glob.glob(data_dir + "/" + folder + '/*')
        for i, f in enumerate(files):
            os.rename(f, os.path.join(self.data_dir + '/' + folder, '{}{:04d}.jpg'.format(folder, i + 1)))


# 이미지 파일 좌우 반전
def image_lr_flip(data_dir):
    # target_data_dir = 'data/transfer_data/transfer_data_cropped+poisoning2'
    data_folder = os.listdir(data_dir)
    for folder in data_folder:
        files = glob.glob(data_dir + '/' + folder + '/*')
        for i, f in enumerate(files):
            temp_new_file_name = 'LRFlip_{}{:04d}.png'.format(folder, i + 1)
            image = Image.open(files[i])
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image.save(data_dir + '/' + folder + '/' + temp_new_file_name)


# 이미지 파일 상하 반전
def image_tb_flip(data_dir):
    # target_data_dir = 'data/transfer_data/transfer_data_cropped+poisoning2'
    data_folder = os.listdir(data_dir)
    for folder in data_folder:
        files = glob.glob(data_dir + '/' + folder + '/*')
        for i, f in enumerate(files):
            temp_new_file_name = 'TBFlip_{}{:04d}.png'.format(folder, i + 1)
            image = Image.open(files[i])
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(data_dir + '/' + folder + '/' + temp_new_file_name)
