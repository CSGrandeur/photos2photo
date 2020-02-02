import os
import sys
import numpy as np
import cv2
from hashlib import md5
import pickle
import tools
from concurrent.futures import ProcessPoolExecutor

PROCESS_NUM = 4
BLOCK_FEATURE_SIZE = 8
RET_BLOCK_SIZE = 64
RET_FEATURE_BLOCK_MAT_SIZE = 64
LIMIT_SEARCH = None
COLOR = True

# the size of the short edge of the result image is RET_FEATURE_BLOCK_MAT_SIZE * RET_BLOCK_SIZE


class PhotoCombination:
    def __init__(self):
        pass

    def get_summary(self, photo_path, refresh=False):
        print("starting photo album summary")
        cache_name = md5(photo_path.encode(encoding='utf-8')).hexdigest() + '_' + str(BLOCK_FEATURE_SIZE) + '_' + str(COLOR) + '.pkl'
        if refresh is False and os.path.exists(cache_name):
            with open(cache_name, 'rb') as f:
                photo_summary = pickle.load(f)
        else:

            pr = ProcessPoolExecutor(PROCESS_NUM)
            photo_summary = []
            obj_list = []
            for img_path in tools.walk_images(photo_path):
                obj_ret = pr.submit(tools.get_rect_img, img_path, BLOCK_FEATURE_SIZE, COLOR)
                obj_list.append((img_path, obj_ret))
            pr.shutdown()
            for item in obj_list:
                feature = item[1].result()
                photo_summary.append((item[0], feature))
            with open(cache_name, 'wb') as f:
                pickle.dump(photo_summary, f)
        print("photo album summary finished")
        return photo_summary

    def combination(self, photo_path, target_path, save_path):
        photo_summary = self.get_summary(photo_path)
        if COLOR:
            img = cv2.imread(target_path)
            height, width, _ = img.shape
        else:
            img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
        if height > width:
            f_resize = float(RET_FEATURE_BLOCK_MAT_SIZE * BLOCK_FEATURE_SIZE) / width
        else:
            f_resize = float(RET_FEATURE_BLOCK_MAT_SIZE * BLOCK_FEATURE_SIZE) / height
        img = cv2.resize(img, None, fx=f_resize, fy=f_resize)
        if height > width:
            img = img[:img.shape[0] // BLOCK_FEATURE_SIZE * BLOCK_FEATURE_SIZE]
        else:
            img = img[:, img.shape[1] // BLOCK_FEATURE_SIZE * BLOCK_FEATURE_SIZE]

        if COLOR:
            ret_img = np.zeros((
                img.shape[0] * RET_BLOCK_SIZE // BLOCK_FEATURE_SIZE,
                img.shape[1] * RET_BLOCK_SIZE // BLOCK_FEATURE_SIZE,
                3), np.uint8)
        else:
            ret_img = np.zeros((
                img.shape[0] * RET_BLOCK_SIZE // BLOCK_FEATURE_SIZE,
                img.shape[1] * RET_BLOCK_SIZE // BLOCK_FEATURE_SIZE
            ), np.uint8)
        pr = ProcessPoolExecutor(PROCESS_NUM)
        obj_list = []
        print("starting block searching")
        for i in range(0, img.shape[0] // BLOCK_FEATURE_SIZE):
            for j in range(0, img.shape[1] // BLOCK_FEATURE_SIZE):
                if COLOR:
                    block_search = img[
                       i * BLOCK_FEATURE_SIZE: (i + 1) * BLOCK_FEATURE_SIZE,
                       j * BLOCK_FEATURE_SIZE: (j + 1) * BLOCK_FEATURE_SIZE,
                       :
                    ]
                else:
                    block_search = img[
                       i * BLOCK_FEATURE_SIZE: (i + 1) * BLOCK_FEATURE_SIZE,
                       j * BLOCK_FEATURE_SIZE: (j + 1) * BLOCK_FEATURE_SIZE
                    ]

                obj_ret = pr.submit(tools.search_block, block_search, photo_summary, RET_BLOCK_SIZE, LIMIT_SEARCH, COLOR)
                # obj_ret = tools.search_block(block_search, photo_summary, RET_BLOCK_SIZE, LIMIT_SEARCH)
                # print(i, j)
                obj_list.append((i, j, obj_ret))

        pr.shutdown()
        print("block searching finished")
        for item in obj_list:
            i, j, block_obj = item
            block = block_obj.result()
            # block = block_obj
            # print(i, j)
            if COLOR:
                ret_img[
                   i * RET_BLOCK_SIZE: (i + 1) * RET_BLOCK_SIZE,
                   j * RET_BLOCK_SIZE: (j + 1) * RET_BLOCK_SIZE,
                   :
                ] = block
            else:
                ret_img[
                   i * RET_BLOCK_SIZE: (i + 1) * RET_BLOCK_SIZE,
                   j * RET_BLOCK_SIZE: (j + 1) * RET_BLOCK_SIZE
                ] = block
        cv2.imwrite(save_path, ret_img)
        print("image saved to %s" % save_path)


if __name__ == '__main__':

    print(sys.argv)
    pc = PhotoCombination()
    photo_path = sys.argv[1]
    target_path = sys.argv[2]
    save_path = sys.argv[3]
    pc.combination(photo_path, target_path, save_path)