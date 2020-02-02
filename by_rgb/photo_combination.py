import sys
import numpy as np
import cv2
from collections import deque
import photo_summary
from concurrent.futures import ProcessPoolExecutor
from config import *

rgb_res_cache = {}
def search_rgb(rgb, color_map):
    """
    搜索符合RGB条件的图片
    """
    q = deque()
    q.append(rgb)
    vis = set()
    vis.add(rgb)
    if rgb in rgb_res_cache and len(rgb_res_cache[rgb]) > 0:
        return rgb_res_cache[rgb][np.random.randint(0, len(rgb_res_cache[rgb]))]
    else:
        rgb_res_cache[rgb] = []
    while len(q) > 0:
        R, G, B = q.popleft()
        if len(color_map[R][G][B]) > 0:
            # file_path = color_map[R][G][B][np.random.randint(0, len(color_map[R][G][B]))]
            rgb_res_cache[rgb] += color_map[R][G][B]
            if len(rgb_res_cache[rgb]) > RGB_CANDIDATE:
                return rgb_res_cache[rgb][np.random.randint(0, len(rgb_res_cache[rgb]))]
        for dx in [0, -1, 1]:
            for dy in [0, -1, 1]:
                for dz in [0, -1, 1]:
                    Rcur = R + dx
                    Gcur = G + dy
                    Bcur = B + dz
                    if (Rcur, Gcur, Bcur) not in vis and 0 <= Rcur < RGB_SCALE and 0 <= Gcur < RGB_SCALE and 0 <= Bcur < RGB_SCALE:
                        vis.add((Rcur, Gcur, Bcur))
                        q.append((Rcur, Gcur, Bcur))

def generate_img(color_map, target_img):
    target_img = (target_img.astype(np.float32) * RGB_SCALE / 256.0).astype(np.uint8)
    ret_img = np.zeros((target_img.shape[0] * BLOCK_SIZE, target_img.shape[1] * BLOCK_SIZE, 3), np.uint8)

    pr = ProcessPoolExecutor(PROCESS_NUM)
    obj_list = []
    for i in range(target_img.shape[0]):
        for j in range(target_img.shape[1]):
            B, G, R = tuple([k for k in target_img[i, j]])
            obj_ret = pr.submit(search_rgb, (R, G, B), color_map)
            # obj_ret = search_rgb((R, G, B), color_map)
            # print(i, j, obj_ret)
            obj_list.append((i, j, obj_ret))

    pr.shutdown()
    for item in obj_list:
        i, j, block_obj = item
        img_file = block_obj.result()
        # img_file = block_obj
        # print(img_file)
        block = photo_summary.get_rect_img(cv2.imread(img_file))
        block = cv2.resize(block, (BLOCK_SIZE, BLOCK_SIZE))
        ret_img[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE, j * BLOCK_SIZE: (j + 1) * BLOCK_SIZE, :] = block[...]

    return ret_img


def combination(photo_path, target_path, save_path):
    print("making summary")
    color_map = photo_summary.get_summary(photo_path)
    print("summary color map finished. start generation")
    img = cv2.imread(target_path)
    height, width, _ = img.shape
    if height > width:
        f_resize = float(RET_SIZE) / width
    else:
        f_resize = float(RET_SIZE) / height
    img = cv2.resize(img, None, fx=f_resize, fy=f_resize)
    ret_img = generate_img(color_map, img)
    print("generation finished. save to %s" % save_path)
    cv2.imwrite(save_path, ret_img)


if __name__ == '__main__':
    print(sys.argv)
    photo_path = sys.argv[1]
    target_path = sys.argv[2]
    save_path = sys.argv[3]
    combination(photo_path, target_path, save_path)