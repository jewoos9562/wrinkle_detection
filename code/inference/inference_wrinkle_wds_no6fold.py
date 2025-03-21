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

def save_results(
    output_mask,
    original_image,
    original_size,
    filename,
    mask_640_dir,
    mask_orig_dir,
    overlay_640_dir,
    overlay_orig_dir
):
    # 1. 마스크 - 640x640
    mask_640 = cv2.resize(output_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
    mask_640_bin = (mask_640 > 0.5).astype(np.uint8) * 255
    path_mask_640 = os.path.join(mask_640_dir, filename)
    cv2.imwrite(path_mask_640, mask_640_bin)

    # 2. 마스크 - 원본 사이즈
    mask_orig = cv2.resize(output_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    mask_orig_bin = (mask_orig > 0.5).astype(np.uint8) * 255
    path_mask_orig = os.path.join(mask_orig_dir, filename)
    cv2.imwrite(path_mask_orig, mask_orig_bin)

    # 3. 오버레이 - 640x640
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    image_640 = cv2.resize(original_bgr, (640, 640))
    overlay_640 = image_640.copy()
    overlay_640[mask_640 > 0.5] = [255, 0, 0]
    blended_640 = cv2.addWeighted(image_640, 0.7, overlay_640, 0.3, 0)
    path_overlay_640 = os.path.join(overlay_640_dir, filename)
    cv2.imwrite(path_overlay_640, blended_640)

    # 4. 오버레이 - 원본 사이즈
    overlay_orig = original_bgr.copy()
    overlay_orig[mask_orig > 0.5] = [255, 0, 0]
    blended_orig = cv2.addWeighted(original_bgr, 0.7, overlay_orig, 0.3, 0)
    path_overlay_orig = os.path.join(overlay_orig_dir, filename)
    cv2.imwrite(path_overlay_orig, blended_orig)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # Paths
    path_src = '/root/skin/wrinkle/dataset/test/images'
    path_ttr = '/root/skin/wrinkle/dataset/test/textures'
    # output_dir = '/root/skin/wrinkle/dataset/test/results/output_masks_3'
    # overlay_dir = '/root/skin/wrinkle/dataset/test/results/output_overlays_3'

    mask_640_dir="/root/skin/wrinkle/dataset/test/results/output_masks_2_640"
    mask_orig_dir="/root/skin/wrinkle/dataset/test/results/output_masks_2"
    overlay_640_dir="/root/skin/wrinkle/dataset/test/results/output_overlays_2_640"
    overlay_orig_dir="/root/skin/wrinkle/dataset/test/results/output_overlays_2"

    os.makedirs(mask_640_dir, exist_ok=True)
    os.makedirs(mask_orig_dir,exist_ok=True)
    os.makedirs(overlay_640_dir, exist_ok=True)
    os.makedirs(overlay_orig_dir, exist_ok=True)
    
    # Load model
    model = UNet_texture_front_ds(4, 2).to(device)
    fns_mdl = '/root/skin/wrinkle/saved_model/20250319_0/model_epoch_144_jsi_0.3236.pth'
    #fns_mdl='/root/skin/wrinkle/saved_model/WRINKLE_WDS/model_epoch_104_jsi_0.3293.pth'
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
        #save_results(output_mask[0], original_image, original_size, filename, output_dir, overlay_dir)
        save_results(
                output_mask[0],
                original_image,
                original_size,
                filename,
                mask_640_dir= mask_640_dir,
                mask_orig_dir= mask_orig_dir,
                overlay_640_dir= overlay_640_dir,
                overlay_orig_dir=overlay_orig_dir
        )


    print("Inference completed. Masks and overlay images saved.")
