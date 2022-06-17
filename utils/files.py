# 폴더 내부 파일 세기
import os
import shutil
import random


def counts_files(data_dir):
    """
    지정한 data directory 전체 내부에 존재하는 file 개수만 세기 위한 함수
    :param data_dir: 탐색할 data directory
    :return: data directory 내부에 존재하는 file의 개수

    Example :
    file_counts = counts_files(data_dir)
    """
    data_folder = os.listdir(data_dir)
    counts_list = []

    for folder in data_folder:
        total_files = 0
        temp_path = os.path.join(data_dir, folder)
        for dir_path, dir_names, file_names in os.walk(temp_path):
            for files in file_names:
                total_files += 1
        counts_list.append(total_files)
    return counts_list


def delete_fewer_folder(data_dir, number):
    """
    Parent directory 내부의 각 directory에서 파일의 개수가 n개 미만일 때 해당 directory를 삭제하는 함수
    :param data_dir: Parent directory 입력 -> 내부 directory 전체 확인함
    :param number: 삭제 기준이 되는 file

    Example :
    delete_fewer_folder(data_dir)
    """
    data_folder = os.listdir(data_dir)

    for folder in data_folder:
        total_files = 0
        temp_path = os.path.join(data_dir, folder)
        for dir_path, dir_names, file_names in os.walk(temp_path):
            for files in file_names:
                total_files += 1
        if total_files < number:
            shutil.rmtree(temp_path)


def search_move(dir_name):
    """
    하위 폴더의 파일을 한 단계 상위 폴더로 이동
    :param dir_name:
    Example : (0)Directory Apple -> (1)directory : red , (2)directory : green
              (1),(2) inner files -> (0) move
    search_move(data_dir)
    """
    folder_list = os.listdir(dir_name)
    for folder in folder_list:
        next_dir = os.path.join(dir_name, folder)
        if os.path.isdir(next_dir):
            search_move(next_dir)
        else:
            s = os.path.split(next_dir)
            p = os.path.split(s[0])
            os.rename(next_dir, p[0] + '/' + s[1])


def drop_empty_folders(directory):
    """
    빈 폴더를 찾아서 삭제
    :param directory:
    """

    for dir_path, dir_names, file_names in os.walk(directory, topdown=False):
        if not dir_names and not file_names:
            os.rmdir(dir_path)


def search_remove_file(dir_name, number):
    """
    n개의 파일만 남기고 삭제하기 위한 함수, 삭제되는 파일은 random.choice에 의해 결정
    :param dir_name: Directory
    :param number: 남길 파일의 개수
    """
    folder_list = os.listdir(dir_name)
    for folder in folder_list:
        temp_dir = os.path.join(dir_name, folder)
        while len(os.listdir(temp_dir)) > number:
            os.remove(temp_dir + '/' + random.choice(os.listdir(temp_dir)))


# n개의 파일 옮기기
def move_data(source_dir, target_dir, number):
    """
    부모 디렉토리에서 각 자식 디렉토리의 n개 파일만 새로운 directory에 옮기기 위한 함수
    옮겨진 파일은 기존 디렉토리에서 삭제된다. -> training, test 데이터를 나눌때 사용
    :param source_dir: Parent directory
    :param target_dir: 옮겨질 파일이 저장될 폴더
    :param number: 옮길 파일의 개수
    """
    folder_list = os.listdir(source_dir)
    for folder in folder_list:
        temp_dir = os.path.join(source_dir, folder)
        os.mkdir(target_dir + '/' + folder)
        while len(os.listdir(temp_dir)) > number:
            shutil.move(temp_dir + '/' + random.choice(os.listdir(temp_dir)), os.path.join(target_dir + '/' + folder))
