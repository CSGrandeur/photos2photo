import os
import numpy as np
import cv2
import pickle
from hashlib import md5
from collections import deque
from concurrent.futures import ProcessPoolExecutor


def get_rect_img(img):
    """
    取图像中间部分的正方形
    """
    height, width, _ = img.shape
    if height > width:
        img = img[0:width, :, :]
    else:
        start = abs(width - height) >> 1
        img = img[:, start: start + height, :]
    return img


def get_img_rgb(file_path):
    """
    统计图像取中间正方形区域的平均RGB
    """
    img = cv2.imread(file_path).astype(np.float32)
    img = get_rect_img(img)
    img = cv2.resize(img, (256, 256))
    B = int(img[..., 0].mean())
    G = int(img[..., 1].mean())
    R = int(img[..., 2].mean())
    return R, G, B


def summary(photo_path):
    """
    预处理影集，统计色彩分布
    """
    color_map = [[[[] for _ in range(256)] for _ in range(256)] for _ in range(256)]
    dq = deque()
    dq.append(photo_path)
    cnt = 0
    pr = ProcessPoolExecutor(4)
    obj_list = []
    while len(dq) > 0:
        cur_path = dq.pop()
        for file in os.listdir(cur_path):
            file_path = os.path.join(cur_path, file)
            if os.path.isdir(file_path):
                dq.append(file_path)
            elif file[-4:].lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                cnt += 1
                print("%06d %s" % (cnt, file_path))
                obj_ret = pr.submit(get_img_rgb, file_path)
                obj_list.append((obj_ret, file_path))

                # R, G, B = get_img_rgb(file_path)
                # color_map[R][G][B].append(file_path)

    pr.shutdown()
    for item in obj_list:
        R, G, B = item[0].result()
        color_map[R][G][B].append(item[1])
    return color_map


def get_summary(photo_path, refresh=False):
    cache_name = md5(photo_path.encode(encoding='utf-8')).hexdigest() + '.pkl'
    if refresh is False and os.path.exists(cache_name):
        with open(cache_name, 'rb') as f:
            color_map = pickle.load(f)
    else:
        color_map = summary(photo_path)
        with open(cache_name, 'wb') as f:
            pickle.dump(color_map, f)
    return color_map


