import os
import sys
import random
import numpy as np
import cv2
from collections import deque


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


def get_rect_img(img_path, size=32, color=True):
    """
    取图像中间部分的正方形
    """
    if color:
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
    block_1 = block_1.astype(np.float32)
    block_2 = block_2.astype(np.float32)
    if len(block_1.shape) == 2:
        block_1 = cv2.dct(block_1)
        block_2 = cv2.dct(block_2)
    else:
        for i in range(3):
            block_1[..., i] = cv2.dct(block_1[..., i])
            block_2[..., i] = cv2.dct(block_2[..., i])
    # return np.linalg.norm(cv2.dct(block_1.astype(np.float32)) - cv2.dct(block_2.astype(np.float32)))
    return np.linalg.norm(block_1.astype(np.float32) / 255.0 - block_2.astype(np.float32) / 255.0)


def search_block(block_s, photo_summary, ret_block_size, limit=None, color=True):
    min_distance = 1e9
    photo_path = ''
    datalen = len(photo_summary)
    if limit is not None:
        random.shuffle(photo_summary)
    for i in range(datalen if limit is None else limit):
        block_obj = photo_summary[i]
        block = block_obj[1]
        dis = block_distance(block_s, block)
        if dis < min_distance:
            min_distance = dis
            photo_path = block_obj[0]
    photo_ret = get_rect_img(photo_path, ret_block_size, color)
    return photo_ret
