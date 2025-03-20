import torch
import torch.onnx
from models.unet_model import UNet_texture_front_ds

# 모델 초기화
n_channels = 4  # 입력 이미지 채널 수 (예: RGB는 3, Grayscale은 1)
n_classes = 2   # 출력 채널 수 (예: 이진 분할의 경우 1, 다중 클래스의 경우 클래스 수 지정)

model = UNet_texture_front_ds(n_channels, n_classes)
model.eval()  # 평가 모드로 설정

# 더미 입력 데이터 생성 (배치 크기 1, 채널 수 3, 256x256 이미지)
dummy_input = torch.randn(1, n_channels, 640, 640)

# ONNX 변환 및 저장
onnx_path = "/root/skin/wrinkle/WeightedDeepSupervision/unet_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # 모델 가중치 포함
    opset_version=11,    # ONNX opset 버전 지정
    do_constant_folding=True,  # 상수 폴딩 최적화
    input_names=["input"],   # 입력 텐서 이름
    output_names=["output"], # 출력 텐서 이름
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 동적 배치 크기 지원
)

print(f"ONNX 모델이 {onnx_path}에 저장되었습니다.")
