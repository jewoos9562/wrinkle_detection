# 🏆 Wrinkle Detection

## 📌 프로젝트 개요
Wrinkle Detection은 U-Net 기반의 **Weighted Deep Supervision** 모델을 활용하여 **주름 영역을 감지하는 딥러닝 기반 모델**입니다.  

---

## 📂 폴더 구조
```
/wrinkle
|-- code
|   |-- WeightedDeepSupervision
|   |-- export
|   |-- inference
|   |-- preprocess
|   `-- train
|-- dataset
|   |-- test
|   `-- train
|-- requirements.txt
`-- saved_model
    |-- 20250318_0
    `-- 20250319_0
```

---

## 🚀 사용 방법

### **1️⃣ 환경 설정**
먼저, 필요한 라이브러리를 설치합니다.

```
pip install -r requirements.txt
```

---

### **2️⃣ 학습 데이터 전처리**
1. 논문에 따르면, 랜드마크를 기반으로 얼굴을 추출하므로 Mediapipe를 활용하여 얼굴 영역을 crop

```bash
python code/preprocess/crop_face.py
```
<br>
2. Crop된 얼굴에서 Texture파일 생성과 이미지 Resize(640 * 640)진행 및 Annotation으로부터 Mask 생성

```bash
python code/preprocess/gen_texture.py
```
<br>
3. 논문의 방법론에 따라 GT를 생성

```bash
python code/preprocess/gen_groundtruth.py
```
---

### **3️⃣ 모델 학습**
모델을 학습하려면 다음 명령어를 실행합니다.

```bash
python code/train/train_wrinkle_wds_no6fold.py
```

- 여러 Hyperparameter는 경우에 따라 수정할 것 (현재 논문 기반) 

✔️ **학습이 완료되면 `saved_model/` 폴더에 가중치(`.pth`)가 저장**

---

### **4️⃣ 모델 추론**
논문의 방법론에 따르면, Input으로 Resized Image (640 * 640 * 3) 과 Texture Map (640 * 640 * 1)가 Concat 되어 들어감
<br>
따라서 Texture Map 만을 추출하는 과정이 필요
```bash
python code/inference/gen_just_texture.py
```
<br>
이후, Concat하여 추론에 활용

```bash
python code/inference/inference_wrinkle_wds_no6fold.py
```
---
