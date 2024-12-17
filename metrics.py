import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def calculate_metrics(generated_frames_paths, reference_frames_paths):
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    
    # Convert the LPIPS model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

    for gen_frame_path, ref_frame_path in zip(generated_frames_paths, reference_frames_paths):
        # Read images using PIL
        gen_frame = Image.open(gen_frame_path).convert("RGB")
        ref_frame = Image.open(ref_frame_path).convert("RGBA")
        
        # Remove the alpha channel of reference image and change the background to white 
        # Create a new image with a white background
        white_bg = Image.new("RGBA", ref_frame.size, (255, 255, 255, 255))
        
        # Composite the image on top of the white background
        ref_frame = Image.alpha_composite(white_bg, ref_frame)

        # Convert to RGB and save the result
        ref_frame = ref_frame.convert("RGB")
        
        ref_frame_array = np.array(ref_frame)
        gen_frame_array = np.array(gen_frame)
        
        # Calculate PSNR and SSIM (which use NumPy)
        psnr_vals.append(psnr(ref_frame_array, gen_frame_array))
        ssim_vals.append(ssim(ref_frame_array, gen_frame_array, multichannel=True, win_size=3, channel_axis=-1))
            
        # Convert the frames to PyTorch tensors and move them to the same device as the model
        gen_frame_tensor = ToTensor()(gen_frame)  # Shape: (1, C, H, W)
        ref_frame_tensor = ToTensor()(ref_frame)  # Shape: (1, C, H, W)
        
        # Ensure tensors have the correct shape for LPIPS: (N, C, H, W)
        gen_frame_tensor = gen_frame_tensor.unsqueeze(0)  # Add batch dimension
        ref_frame_tensor = ref_frame_tensor.unsqueeze(0)

        # Move tensors to GPU if available
        gen_frame_tensor = gen_frame_tensor.to(device)
        ref_frame_tensor = ref_frame_tensor.to(device)

        # Calculate LPIPS
        lpips_vals.append(lpips(gen_frame_tensor, ref_frame_tensor))

   #return psnr_vals, ssim_vals, lpips_vals
    metrics_dict = {
        'psnr': psnr_vals,
        'ssim': ssim_vals,
        'lpips': lpips_vals,
        'frame_count': len(generated_frames_paths)
    }
    return metrics_dict

def calculate_avge(frames):
    return np.mean(frames) 
