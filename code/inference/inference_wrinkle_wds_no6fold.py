import torch
from torch.autograd import Variable
import sys, os
import numpy as np
import cv2

wd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "WeightedDeepSupervision"))
sys.path.append(wd_path)
from models.unet_model import UNet_texture_front_ds

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preprocess_image(image_path, size=(640, 640)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, size).astype(np.float32) / 255.0
    
    # Apply normalization (same as Dataset_Wrinkle_WDS)
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std = np.array([0.25, 0.25, 0.25]).reshape(1, 1, 3)
    image_resized = (image_resized - mean) / std
    
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image_tensor, image, original_size

def preprocess_texture(texture_path, size=(640, 640)):
    texture = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
    texture_resized = cv2.resize(texture, size).astype(np.float32) / 255.0
    
    # Apply grayscale normalization (same as Dataset_Wrinkle_WDS)
    mean = 0.5
    std = 0.25
    texture_resized = (texture_resized - mean) / std
    
    texture_tensor = torch.tensor(texture_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return texture_tensor

def inference(model, image_tensor, texture_tensor, device):
    image_tensor = Variable(image_tensor).to(device)
    texture_tensor = Variable(texture_tensor).to(device)
    softmax_2d = torch.nn.Softmax2d()

    with torch.no_grad():
        out_1, _, _, _ = model(image_tensor, texture_tensor)
        score = softmax_2d(out_1).cpu().numpy()
    
    return score[:, 1, :, :]

def save_results(output_mask, original_image, original_size, filename, output_dir, overlay_dir):
    output_mask_resized = cv2.resize(output_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # 마스크 저장
    output_path = os.path.join(output_dir, filename.replace('.png', '_mask.png'))
    cv2.imwrite(output_path, (output_mask_resized > 0.5).astype(np.uint8) * 255)

    # 원본 이미지를 다시 BGR로 변환
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Overlay 적용
    overlay = original_image_bgr.copy()
    mask_indices = output_mask_resized > 0.5
    overlay[mask_indices] = [255, 0, 0]  # 빨간색 (BGR)
    blended = cv2.addWeighted(original_image_bgr, 0.7, overlay, 0.3, 0)

    # 오버레이 저장
    overlay_path = os.path.join(overlay_dir, filename.replace('.png', '_overlay.png'))
    cv2.imwrite(overlay_path, blended)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # Paths
    path_src = '/root/skin/wrinkle/dataset/test_images'
    path_ttr = '/root/skin/wrinkle/dataset/test_textures'
    output_dir = '/root/skin/wrinkle/dataset/test_output_masks_2'
    overlay_dir = '/root/skin/wrinkle/dataset/test_output_overlays_2'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Load model
    model = UNet_texture_front_ds(4, 2).to(device)
    #fns_mdl = '/root/skin/save_model/WRINKLE_WDS/model_epoch_156_jsi_0.3335.pth'
    fns_mdl='/root/skin/wrinkle/saved_model/20250319_0/model_epoch_144_jsi_0.3236.pth'
    model.load_state_dict(torch.load(fns_mdl, map_location=device), strict=True)
    model.eval()

    # Process each image in folder
    for filename in os.listdir(path_src):
        image_path = os.path.join(path_src, filename)
        texture_path = os.path.join(path_ttr, filename)
        
        if not os.path.exists(texture_path):
            print(f"Skipping {filename}, texture not found.")
            continue
        
        image_tensor, original_image, original_size = preprocess_image(image_path)
        texture_tensor = preprocess_texture(texture_path)
        
        output_mask = inference(model, image_tensor, texture_tensor, device)
        save_results(output_mask[0], original_image, original_size, filename, output_dir, overlay_dir)
    
    print("Inference completed. Masks and overlay images saved.")
