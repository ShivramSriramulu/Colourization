"""
Utility functions for Image Colorization project

This module contains helper functions for data preprocessing, visualization,
and model evaluation.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchvision import transforms

def load_image_paths(data_dir, extensions=['*.jpg', '*.jpeg', '*.png']):
    """Load image paths from directory with specified extensions"""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
        paths.extend(glob.glob(os.path.join(data_dir, ext.upper())))
    return sorted(paths)

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return transform(img)

def rgb_to_lab(rgb_image):
    """Convert RGB image to LAB color space"""
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
    
    lab_image = rgb2lab(rgb_image).astype("float32")
    return torch.from_numpy(lab_image).permute(2, 0, 1)

def lab_to_rgb(lab_image):
    """Convert LAB image to RGB color space"""
    if isinstance(lab_image, torch.Tensor):
        lab_image = lab_image.permute(1, 2, 0).numpy()
    
    rgb_image = lab2rgb(lab_image)
    return (rgb_image * 255).astype(np.uint8)

def visualize_results(original, colorized, grayscale, save_path=None):
    """Visualize original, grayscale, and colorized images side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(grayscale, cmap='gray')
    axes[1].set_title('Grayscale Input')
    axes[1].axis('off')
    
    axes[2].imshow(colorized)
    axes[2].set_title('Colorized Output')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(original, colorized):
    """Calculate image quality metrics"""
    if isinstance(original, torch.Tensor):
        original = original.permute(1, 2, 0).numpy()
    if isinstance(colorized, torch.Tensor):
        colorized = colorized.permute(1, 2, 0).numpy()
    
    # Convert to grayscale for SSIM
    if original.shape[2] == 3:
        original_gray = np.dot(original[..., :3], [0.299, 0.587, 0.114])
    else:
        original_gray = original
    
    if colorized.shape[2] == 3:
        colorized_gray = np.dot(colorized[..., :3], [0.299, 0.587, 0.114])
    else:
        colorized_gray = colorized
    
    # Calculate metrics
    ssim_score = ssim(original_gray, colorized_gray, data_range=255)
    psnr_score = psnr(original, colorized, data_range=255)
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score
    }

def create_color_palette():
    """Create a color palette for visualization"""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    return colors

def save_model_summary(model, save_path):
    """Save model architecture summary"""
    with open(save_path, 'w') as f:
        f.write(str(model))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write(f"\n\nTotal parameters: {total_params:,}")
        f.write(f"\nTrainable parameters: {trainable_params:,}")

def create_training_plots(losses, save_dir):
    """Create and save training plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot generator losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses['gen_loss'], label='Generator Loss')
    plt.plot(losses['gen_l1_loss'], label='L1 Loss')
    plt.title('Generator Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'generator_losses.png'))
    plt.close()
    
    # Plot discriminator losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses['disc_loss'], label='Discriminator Loss')
    plt.plot(losses['disc_real_loss'], label='Real Loss')
    plt.plot(losses['disc_fake_loss'], label='Fake Loss')
    plt.title('Discriminator Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'discriminator_losses.png'))
    plt.close()

def validate_image_paths(image_paths):
    """Validate that all image paths exist and are readable"""
    valid_paths = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Invalid image {path}: {e}")
    return valid_paths
