import sys
import numpy as np
import cv2
from collections import deque
import photo_summary

RET_SIZE = 64
BLOCK_SIZE = 32


def search_rgb(rgb, color_map):
    """
    搜索符合RGB条件的图片
    """
    q = deque()
    q.append(rgb)
    vis = set()
    vis.add(rgb)
    while len(q) > 0:
        R, G, B = q.popleft()
        if len(color_map[R][G][B]) > 0:
            file_path = color_map[R][G][B][np.random.randint(0, len(color_map[R][G][B]))]
            return file_path
        for dx in [0, -1, 1]:
            for dy in [0, -1, 1]:
                for dz in [0, -1, 1]:
                    R += dx
                    G += dy
                    B += dz
                    if 0 <= R <= 255 and 0 <= G <= 255 and 0 <= B <= 255 and (R, G, B) not in vis:
                        vis.add((R, G, B))
                        q.append((R, G, B))


def generate_img(color_map, target_img):
    ret_img = np.zeros((target_img.shape[0] * BLOCK_SIZE, target_img.shape[1] * BLOCK_SIZE, 3), np.uint8)
    for i in range(target_img.shape[0]):
        for j in range(target_img.shape[1]):
            B, G, R = tuple([k for k in target_img[i, j]])
            img_file = search_rgb((R, G, B), color_map)
            print("coordinate:", (i, j), "RGB:", (R, G, B), img_file)
            block = photo_summary.get_rect_img(cv2.imread(img_file))
            block = cv2.resize(block, (BLOCK_SIZE, BLOCK_SIZE))
            ret_img[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE, j * BLOCK_SIZE: (j + 1) * BLOCK_SIZE, :] = block[...]

    return ret_img


def combination(photo_path, target_path, save_path):
    color_map = photo_summary.get_summary(photo_path)
    img = cv2.imread(target_path)
    height, width, _ = img.shape
    if height > width:
        f_resize = float(RET_SIZE) / width
    else:
        f_resize = float(RET_SIZE) / height
    img = cv2.resize(img, None, fx=f_resize, fy=f_resize)
    ret_img = generate_img(color_map, img)
    cv2.imwrite(save_path, ret_img)


if __name__ == '__main__':
    print(sys.argv)
    photo_path = sys.argv[1]
    target_path = sys.argv[2]
    save_path = sys.argv[3]
    combination(photo_path, target_path, save_path)