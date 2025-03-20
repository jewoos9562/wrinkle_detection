# 🏆 Wrinkle Detection

## 📌 프로젝트 개요
Wrinkle Detection은 **주름(Wrinkle) 영역을 감지하는 딥러닝 기반 모델**입니다.  
이 프로젝트는 U-Net 기반의 **Weighted Deep Supervision(가중 딥 슈퍼비전)** 모델을 활용하여 **이미지에서 주름을 정확히 세그멘테이션(분할)** 합니다.

---

## 📂 폴더 구조
```
/wrinkle_detection
│── models/                        # 모델 가중치 및 저장
│── dataset/                        # 데이터셋 폴더
│── scripts/                        # 전처리 및 학습 스크립트
│── inference/                      # 추론 스크립트
│── results/                        # 예측 결과 저장 폴더
│── README.md                       # 프로젝트 설명 파일
│── train.py                        # 모델 학습 코드
│── inference.py                    # 모델 추론 코드
│── requirements.txt                 # 필요한 패키지 목록
```

---

## 🚀 사용 방법

### **1️⃣ 환경 설정**
먼저, 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

---

### **2️⃣ 데이터 준비**
데이터셋을 `dataset/` 폴더에 배치하고, 학습을 위해 전처리를 수행합니다.

```bash
python scripts/preprocess.py --data_dir dataset/
```

---

### **3️⃣ 모델 학습**
모델을 학습하려면 다음 명령어를 실행합니다.

```bash
python train.py --epochs 100 --batch_size 16 --lr 0.0001
```

- `--epochs`: 학습 반복 횟수  
- `--batch_size`: 배치 크기  
- `--lr`: 학습률 (learning rate)  

✔️ **학습이 완료되면 `models/` 폴더에 가중치(`.pth`)가 저장됩니다.**

---

### **4️⃣ 모델 추론**
주름 감지를 위해 이미지를 입력하고 추론을 수행합니다.

```bash
python inference.py --image_path test.jpg --model_path models/best_model.pth
```

✔️ 결과는 `results/` 폴더에 저장됩니다.

---

## 📊 모델 구조
Wrinkle Detection 모델은 **U-Net 기반의 Weighted Deep Supervision 기법**을 사용하여 정확도를 향상시킵니다.

```python
import torch
from models.unet_model import UNet_texture_front_ds

model = UNet_texture_front_ds(n_channels=4, n_classes=2)
print(model)
```

📌 **주요 특징**
- RGB 이미지 + 텍스처 이미지를 활용한 **4채널 입력**
- **Softmax 기반의 확률 예측**
- 가중치 손실 적용으로 **정확한 주름 검출 가능**

---

## 🎯 결과 예시
| 원본 이미지 | 예측 마스크 |
|------------|------------|
| ![input](results/example_input.jpg) | ![output](results/example_output.jpg) |

---

## 📌 요구 사항
✅ 이 프로젝트는 **Python 3.8+**에서 실행됩니다.  
✅ 필수 패키지는 `requirements.txt`에 정의되어 있으며, 다음 주요 라이브러리를 포함합니다.

```txt
torch
torchvision
opencv-python
numpy
matplotlib
```

✔️ 패키지 설치:
```bash
pip install -r requirements.txt
```

---

## 🤝 기여 방법
이 프로젝트에 기여하려면 **Fork 후 Pull Request(PR)를 생성**해주세요.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/wrinkle_detection.git
cd wrinkle_detection
git checkout -b feature-branch
git add .
git commit -m "Add new feature"
git push origin feature-branch
```

---

## 📜 라이선스
이 프로젝트는 **MIT 라이선스**를 따릅니다.  
자유롭게 활용하되, 원저작자 표기를 유지해주세요.

---

## 📧 문의
궁금한 점이 있으면 아래로 연락 주세요!  
📩 **Email:** your_email@example.com  
💬 **GitHub Issues:** [프로젝트 이슈 트래커](https://github.com/YOUR_GITHUB_USERNAME/wrinkle_detection/issues)

