#!/usr/bin/env python3
"""
Image Colorization with GAN - Main Entry Point

This script provides a command-line interface for training, evaluating, and running inference
on the image colorization model using a GAN architecture with U-Net generator.
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class Config:
    """Configuration class for model hyperparameters"""
    external_data_size = 10000
    train_size = 8000
    image_size_1 = 256
    image_size_2 = 256
    batch_size = 32
    LeakyReLU_slope = 0.2
    dropout = 0.5
    kernel_size = 4
    stride = 2
    padding = 1
    gen_lr = 2e-4
    disc_lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_l1 = 100
    gan_mode = 'vanilla'
    layers_to_cut = -2
    epochs = 20
    pretrain_lr = 1e-4

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    """Create loss meters for tracking training progress"""
    return {
        'disc_loss_gen': AverageMeter(),
        'disc_loss_real': AverageMeter(),
        'disc_loss': AverageMeter(),
        'loss_G': AverageMeter(),
        'loss_G_L1': AverageMeter()
    }

class ImageDataset(Dataset):
    """Dataset class for loading and preprocessing images"""
    
    def __init__(self, paths, train=True):
        self.paths = paths
        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((Config.image_size_1, Config.image_size_2)),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((Config.image_size_1, Config.image_size_2))
            ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return {'L': L, 'ab': ab}

class UnetBlock(nn.Module):
    """U-Net block for the generator"""
    
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(ni * 2, nf, kernel_size=3, stride=1, padding=1, bias=False)
            down = [downconv]
            up = [uprelu, upsample, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(ni * 2, nf, kernel_size=3, stride=1, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """U-Net generator with ResNet18 backbone"""
    
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, input):
        return self.model(input)

class PatchDiscriminator(nn.Module):
    """Patch-based discriminator"""
    
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1) 
                 for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)
    
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def init_weights(net, init='norm', gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    
    net.apply(init_func)
    return net

def init_model(model, device):
    """Initialize model and move to device"""
    model = init_weights(model)
    return model.to(device)

def load_model(model_path: str) -> Tuple[UnetGenerator, PatchDiscriminator]:
    """Load trained models from checkpoint"""
    logger.info(f"Loading model from {model_path}")
    
    # Initialize models
    gen = UnetGenerator()
    disc = PatchDiscriminator(input_c=3)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'gen_state_dict' in checkpoint:
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
    else:
        # Assume it's just the generator
        gen.load_state_dict(checkpoint)
    
    gen.eval()
    disc.eval()
    
    logger.info("Model loaded successfully")
    return gen, disc

def colorize_image(image_path: str, gen: UnetGenerator, output_path: Optional[str] = None) -> Image.Image:
    """Colorize a single image"""
    logger.info(f"Colorizing image: {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((Config.image_size_1, Config.image_size_2)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    
    # Convert to LAB
    img_array = img_tensor.permute(1, 2, 0).numpy()
    img_lab = rgb2lab(img_array).astype("float32")
    img_lab = torch.from_numpy(img_lab).permute(2, 0, 1)
    
    L = img_lab[[0], ...] / 50. - 1.
    L = L.unsqueeze(0).to(device)
    
    # Generate colorization
    with torch.no_grad():
        fake_ab = gen(L)
    
    # Convert back to RGB
    fake_ab = fake_ab.squeeze(0).cpu()
    fake_ab = fake_ab * 110.
    fake_lab = torch.cat([img_lab[[0]], fake_ab], dim=0).permute(1, 2, 0).numpy()
    fake_rgb = lab2rgb(fake_lab)
    fake_rgb = (fake_rgb * 255).astype(np.uint8)
    
    result = Image.fromarray(fake_rgb)
    
    if output_path:
        result.save(output_path)
        logger.info(f"Colorized image saved to: {output_path}")
    
    return result

def train_model(args):
    """Train the colorization model"""
    logger.info("Starting model training")
    logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    # TODO: Implement training loop
    # This would include:
    # 1. Data loading and preprocessing
    # 2. Model initialization
    # 3. Training loop with loss computation
    # 4. Model checkpointing
    # 5. Validation and metrics tracking
    
    logger.info("Training completed (placeholder implementation)")

def evaluate_model(args):
    """Evaluate the trained model"""
    logger.info("Starting model evaluation")
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Load model
    gen, disc = load_model(args.model_path)
    
    # Process test images
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for img_path in test_dir.glob("*.jpg"):
        logger.info(f"Processing: {img_path}")
        output_path = output_dir / f"colorized_{img_path.name}"
        colorize_image(str(img_path), gen, str(output_path))
    
    logger.info(f"Evaluation completed. Results saved to: {output_dir}")

def run_inference(args):
    """Run inference on a single image"""
    logger.info("Running inference")
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        return
    
    # Load model
    gen, disc = load_model(args.model_path)
    
    # Colorize image
    output_path = args.output_dir if args.output_dir else "colorized_output.jpg"
    colorize_image(args.input_image, gen, output_path)
    
    logger.info(f"Inference completed. Result saved to: {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Image Colorization with GAN")
    parser.add_argument('--mode', choices=['train', 'eval', 'inference'], 
                       default='inference', help='Mode to run')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, help='Directory containing training images')
    parser.add_argument('--test_dir', type=str, default='test_images/', 
                       help='Directory containing test images')
    parser.add_argument('--input_image', type=str, help='Single image for inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='main-model.pt',
                       help='Path to model checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run appropriate mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'eval':
        evaluate_model(args)
    elif args.mode == 'inference':
        run_inference(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
