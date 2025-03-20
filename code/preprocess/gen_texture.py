import cv2
import os
import glob
import numpy as np
import json
from tqdm import tqdm

# 경로 설정
path_src = '/root/skin/wrinkle/dataset/images'  # 원본 이미지 경로
path_dst_texture = '/root/skin/wrinkle/dataset/textures'  # 텍스처 저장 경로
path_dst_resized = '/root/skin/wrinkle/dataset/images_resized'  # 리사이즈된 원본 저장 경로
path_annotations = '/root/skin/wrinkle/dataset/annotations'  # 주름 JSON 파일 경로
path_masks_resized = '/root/skin/wrinkle/dataset/masks_resized'  # 리사이즈된 마스크 저장 경로

# 필요한 폴더 생성
os.makedirs(path_dst_texture, exist_ok=True)
os.makedirs(path_dst_resized, exist_ok=True)
os.makedirs(path_masks_resized, exist_ok=True)

# PNG 이미지 리스트 가져오기
list_png = glob.glob(f'{path_src}/*.png')
list_png.sort()

# 가우시안 블러 커널 설정
kernel1d = cv2.getGaussianKernel(21, 5)
kernel2d = np.outer(kernel1d, kernel1d.transpose())

# 이미지 & 마스크 처리
for fns_src in tqdm(list_png, desc="Processing images"):
    # 파일명 추출 (예: 69023.png → 69023)
    filename = os.path.basename(fns_src).replace(".png", "")
    if filename+'.json' in os.listdir(path_annotations):

        # 저장 경로 설정
        fns_dst_texture = os.path.join(path_dst_texture, f"{filename}.png")  # 텍스처 저장
        fns_dst_resized = os.path.join(path_dst_resized, f"{filename}.png")  # 리사이즈된 원본 저장
        fns_mask_resized = os.path.join(path_masks_resized, f"{filename}.png")  # 리사이즈된 마스크 저장

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

        # 🔹 JSON 파일을 이용해 마스크 생성 (해당 파일이 존재하는 경우)
        json_path = os.path.join(path_annotations, f"{filename}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            # 원본 이미지 크기 확인
            h, w, _ = img_src.shape  # 기존 원본 이미지 크기

            # 빈 마스크 생성 (검은색 배경)
            mask = np.zeros((h, w, 3), dtype=np.uint8)  # 컬러 마스크로 변경

            # JSON에서 주름 좌표 추출 및 마스크 생성
            for shape in data["shapes"]:
                points = np.array(shape["points"], dtype=np.int32)  # 좌표 변환
                cv2.polylines(mask, [points], isClosed=False, color=(255, 255, 255), thickness=2)  # 주름을 흰색(255,255,255)으로 그림

            # 마스크 리사이즈 (640x640)
            mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(fns_mask_resized, mask_resized)  # 리사이즈된 마스크 저장

print("Processing finished. Resized images & masks saved.")