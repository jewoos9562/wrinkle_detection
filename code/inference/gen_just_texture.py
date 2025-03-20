import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# 경로 설정
path_src = '/root/skin/wrinkle/dataset/test_images'  # 원본 이미지 경로
path_dst_texture = '/root/skin/wrinkle/dataset/test_textures'  # 텍스처 저장 경로
path_dst_resized = '/root/skin/wrinkle/dataset/test_images_resized'  # 리사이즈된 원본 저장 경로

# 필요한 폴더 생성
os.makedirs(path_dst_texture, exist_ok=True)
os.makedirs(path_dst_resized, exist_ok=True)

# PNG 이미지 리스트 가져오기
list_png = glob.glob(f'{path_src}/*.png')
list_png.sort()

# 가우시안 블러 커널 설정
kernel1d = cv2.getGaussianKernel(21, 5)
kernel2d = np.outer(kernel1d, kernel1d.transpose())

# 이미지 & 텍스처 처리
for fns_src in tqdm(list_png, desc="Processing images"):
    # 파일명 추출 (예: 69023.png → 69023)
    filename = os.path.basename(fns_src).replace(".png", "")

    # 저장 경로 설정
    fns_dst_texture = os.path.join(path_dst_texture, f"{filename}.png")  # 텍스처 저장
    fns_dst_resized = os.path.join(path_dst_resized, f"{filename}.png")  # 리사이즈된 원본 저장

    # 원본 이미지 로드 및 리사이즈
    img_src = cv2.imread(fns_src, cv2.IMREAD_COLOR)  # 컬러 이미지로 로드
    img_src_resized = cv2.resize(img_src, (640, 640))  # 원본 이미지 리사이즈
    cv2.imwrite(fns_dst_resized, img_src_resized)  # 리사이즈된 원본 저장

    # 텍스처 맵 생성
    img_src_float = np.array(cv2.cvtColor(img_src_resized, cv2.COLOR_BGR2GRAY), dtype=float)  # Grayscale 변환 후 처리
    img_low = cv2.filter2D(img_src_float, -1, kernel2d)
    img_low = np.array(img_low, dtype=float)

    img_div = (img_src_float * 255.) / (img_low + 1.)
    img_div[img_div > 255.] = 255.
    img_div = np.array(img_div, dtype=np.uint8)
    img_div = 1 - img_div

    cv2.imwrite(fns_dst_texture, img_div)  # 텍스처 이미지 저장

print("Processing finished. Resized images & textures saved.")