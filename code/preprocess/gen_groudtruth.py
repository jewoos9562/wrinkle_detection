# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab
import cv2
import os
import glob
from tqdm import tqdm

if __name__ == '__main__':

    # Edit your path
    # path_src = '/root/skin/wrinkle/dataset/train/textures'
    # path_pgt = '/root/skin/wrinkle/dataset/train/masks_resized'
    # path_dst = '/root/skin/wrinkle/dataset/train/GT'

    path_src = '/root/skin/wrinkle/dataset/test/textures'
    path_pgt = '/root/skin/wrinkle/dataset/test/masks_resized'
    path_dst = '/root/skin/wrinkle/dataset/test/GT'


    if os.path.isdir(path_dst) == False:
        os.makedirs(path_dst)

    # Get list of images
    list_png = glob.glob('%s/*.png' % path_src)
    list_png.sort()

    # Get groundtruth
    for fns_src in tqdm(list_png):
        str_src = os.path.basename(fns_src)
        print(str_src)
        if str_src in os.listdir(path_pgt):

            fns_pgt = '%s/%s' % (path_pgt, str_src)
            fns_dst = fns_src.replace(path_src, path_dst)

            print(fns_pgt)

            img_src = cv2.imread(fns_src, cv2.IMREAD_GRAYSCALE)
            img_pgt = cv2.imread(fns_pgt, cv2.IMREAD_GRAYSCALE)
            img_src[img_pgt == 0] = 0

            img_dst = cv2.adaptiveThreshold(255 - img_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
            img_dst = 255 - img_dst
            cv2.imwrite(fns_dst, img_dst)

    print('finished')
