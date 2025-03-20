import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe FaceMesh 모델 로드
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 이미지 로드
image_folder="/root/skin/wrinkle/dataset/original_test_images"
result_folder="/root/skin/wrinkle/dataset/test_images"
os.makedirs(result_folder,exist_ok=True)
image_list=os.listdir(image_folder)

for image_name in image_list:

    image_path=os.path.join(image_folder,image_name)
    #image_path = "/root/skin/wrinkle/dataset/images/69308.png"  # 입력 이미지 경로
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 랜드마크 검출
    results = face_mesh.process(image_rgb)

    # 얼굴 BBox 크롭
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            
            # 랜드마크 좌표 추출
            landmark_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

            # 가장 왼쪽, 오른쪽, 위쪽, 아래쪽 포인트 찾기
            leftmost = min(landmark_points, key=lambda point: point[0])[0]
            rightmost = max(landmark_points, key=lambda point: point[0])[0]
            topmost = min(landmark_points, key=lambda point: point[1])[1]
            bottommost = max(landmark_points, key=lambda point: point[1])[1]


            topmost = max(int(topmost - (bottommost - topmost) * 0.20), 0)
            # BBox 크롭
            cropped_face = image[topmost:bottommost, leftmost:rightmost]

            # 크롭된 이미지 저장
            output_crop_path=os.path.join(result_folder,image_name)
            #output_crop_path = "/root/skin/face_landmarks.jpg"
            cv2.imwrite(output_crop_path, cropped_face)
            print(f"크롭된 얼굴 이미지가 저장됨: {output_crop_path}")
