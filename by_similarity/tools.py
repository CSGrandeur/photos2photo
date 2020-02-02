import os
import sys
import random
import numpy as np
import cv2
from collections import deque
from config import *

def walk_images(photo_path):
    """
    遍历目录返回图片
    """
    dq = deque()
    dq.append(photo_path)
    while len(dq) > 0:
        cur_path = dq.pop()
        for file in os.listdir(cur_path):
            file_path = os.path.join(cur_path, file)
            if os.path.isdir(file_path):
                dq.append(file_path)
            elif file[-4:].lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                yield file_path


def get_rect_img(img_path, size=32):
    """
    取图像中间部分的正方形
    """
    if COLOR:
        img = cv2.imread(img_path)
        height, width, _ = img.shape
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
    if height > width:
        img = img[0:width, ...]
    else:
        start = abs(width - height) >> 1
        img = img[:, start: start + height]
    img = cv2.resize(img, (size, size))
    return img


def block_distance(block_1, block_2):
    # block_1 = block_1.astype(np.float32)
    # block_2 = block_2.astype(np.float32)
    # if len(block_1.shape) == 2:
    #     block_1 = cv2.dct(block_1)
    #     block_2 = cv2.dct(block_2)
    # else:
    #     for i in range(3):
    #         block_1[..., i] = cv2.dct(block_1[..., i])
    #         block_2[..., i] = cv2.dct(block_2[..., i])

    return np.linalg.norm(block_1.astype(np.float32) / 255.0 - block_2.astype(np.float32) / 255.0)


def search_block(block_s, photo_summary):
    datalen = len(photo_summary)
    if LIMIT_SEARCH is not None:
        random.shuffle(photo_summary)
    res_list = []
    for i in range(datalen if LIMIT_SEARCH is None else LIMIT_SEARCH):
        block_obj = photo_summary[i]
        block = block_obj[1]
        dis = block_distance(block_s, block)
        res_list.append((dis, i))
    res_list = sorted(res_list)
    photo_path = photo_summary[res_list[np.random.randint(0, min(len(res_list), SIMILAR_CANDIDATE))][1]][0]
    photo_ret = get_rect_img(photo_path, RET_BLOCK_SIZE)
    return photo_ret
