"""
Evaluation metrics for Image Colorization project

This module provides comprehensive evaluation metrics for assessing
the quality of colorized images and model performance.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import seaborn as sns

class ColorizationMetrics:
    """Comprehensive metrics calculator for image colorization"""
    
    def __init__(self):
        self.metrics_history = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'color_accuracy': [],
            'perceptual_loss': []
        }
    
    def calculate_psnr(self, original: np.ndarray, colorized: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(original, colorized, data_range=255)
    
    def calculate_ssim(self, original: np.ndarray, colorized: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Convert to grayscale for SSIM calculation
        if original.shape[2] == 3:
            original_gray = np.dot(original[..., :3], [0.299, 0.587, 0.114])
        else:
            original_gray = original
            
        if colorized.shape[2] == 3:
            colorized_gray = np.dot(colorized[..., :3], [0.299, 0.587, 0.114])
        else:
            colorized_gray = colorized
            
        return ssim(original_gray, colorized_gray, data_range=255)
    
    def calculate_mse(self, original: np.ndarray, colorized: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return mse(original, colorized)
    
    def calculate_color_accuracy(self, original: np.ndarray, colorized: np.ndarray) -> float:
        """Calculate color accuracy in LAB space"""
        # Convert to LAB color space
        original_lab = rgb2lab(original / 255.0)
        colorized_lab = rgb2lab(colorized / 255.0)
        
        # Calculate color difference (Euclidean distance in LAB space)
        color_diff = np.sqrt(np.sum((original_lab - colorized_lab) ** 2, axis=2))
        
        # Return mean color accuracy (lower is better)
        return np.mean(color_diff)
    
    def calculate_perceptual_loss(self, original: np.ndarray, colorized: np.ndarray) -> float:
        """Calculate perceptual loss using VGG features (placeholder)"""
        # This would typically use a pre-trained VGG network
        # For now, we'll use a simple approximation
        return np.mean(np.abs(original - colorized))
    
    def calculate_all_metrics(self, original: np.ndarray, colorized: np.ndarray) -> Dict[str, float]:
        """Calculate all available metrics"""
        metrics = {
            'psnr': self.calculate_psnr(original, colorized),
            'ssim': self.calculate_ssim(original, colorized),
            'mse': self.calculate_mse(original, colorized),
            'color_accuracy': self.calculate_color_accuracy(original, colorized),
            'perceptual_loss': self.calculate_perceptual_loss(original, colorized)
        }
        
        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all evaluations"""
        return {
            key: np.mean(values) if values else 0.0
            for key, values in self.metrics_history.items()
        }
    
    def reset_history(self):
        """Reset metrics history"""
        for key in self.metrics_history:
            self.metrics_history[key] = []
    
    def plot_metrics_history(self, save_path: Optional[str] = None):
        """Plot metrics history over time"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Metrics History', fontsize=16)
        
        metrics_list = list(self.metrics_history.keys())
        
        for i, metric in enumerate(metrics_list):
            row = i // 3
            col = i % 3
            
            if self.metrics_history[metric]:
                axes[row, col].plot(self.metrics_history[metric])
                axes[row, col].set_title(f'{metric.upper()} History')
                axes[row, col].set_xlabel('Evaluation Step')
                axes[row, col].set_ylabel(metric.upper())
                axes[row, col].grid(True)
            else:
                axes[row, col].text(0.5, 0.5, f'No {metric} data', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{metric.upper()} History')
        
        # Remove empty subplot if needed
        if len(metrics_list) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def calculate_batch_metrics(original_batch: torch.Tensor, colorized_batch: torch.Tensor) -> Dict[str, float]:
    """Calculate metrics for a batch of images"""
    metrics_calc = ColorizationMetrics()
    
    # Convert tensors to numpy arrays
    if isinstance(original_batch, torch.Tensor):
        original_batch = original_batch.cpu().numpy()
    if isinstance(colorized_batch, torch.Tensor):
        colorized_batch = colorized_batch.cpu().numpy()
    
    # Ensure proper format (B, C, H, W) -> (B, H, W, C)
    if original_batch.shape[1] == 3:
        original_batch = np.transpose(original_batch, (0, 2, 3, 1))
    if colorized_batch.shape[1] == 3:
        colorized_batch = np.transpose(colorized_batch, (0, 2, 3, 1))
    
    # Scale to 0-255 range
    original_batch = (original_batch * 255).astype(np.uint8)
    colorized_batch = (colorized_batch * 255).astype(np.uint8)
    
    # Calculate metrics for each image in batch
    batch_metrics = []
    for i in range(original_batch.shape[0]):
        metrics = metrics_calc.calculate_all_metrics(
            original_batch[i], colorized_batch[i]
        )
        batch_metrics.append(metrics)
    
    # Average across batch
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
    
    return avg_metrics

def evaluate_model_performance(model_outputs: List[np.ndarray], 
                             ground_truth: List[np.ndarray]) -> Dict[str, float]:
    """Evaluate model performance on a dataset"""
    metrics_calc = ColorizationMetrics()
    
    all_metrics = []
    for pred, gt in zip(model_outputs, ground_truth):
        metrics = metrics_calc.calculate_all_metrics(gt, pred)
        all_metrics.append(metrics)
    
    # Calculate average metrics
    final_metrics = {}
    for key in all_metrics[0].keys():
        final_metrics[key] = np.mean([m[key] for m in all_metrics])
        final_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    
    return final_metrics

def generate_metrics_report(metrics: Dict[str, float], save_path: Optional[str] = None) -> str:
    """Generate a formatted metrics report"""
    report = "Image Colorization Model Performance Report\n"
    report += "=" * 50 + "\n\n"
    
    # Quality metrics
    report += "Quality Metrics:\n"
    report += f"  PSNR: {metrics.get('psnr', 0):.2f} dB\n"
    report += f"  SSIM: {metrics.get('ssim', 0):.4f}\n"
    report += f"  MSE: {metrics.get('mse', 0):.4f}\n"
    report += f"  Color Accuracy: {metrics.get('color_accuracy', 0):.2f}\n"
    report += f"  Perceptual Loss: {metrics.get('perceptual_loss', 0):.4f}\n\n"
    
    # Standard deviations if available
    if 'psnr_std' in metrics:
        report += "Standard Deviations:\n"
        report += f"  PSNR: ±{metrics.get('psnr_std', 0):.2f} dB\n"
        report += f"  SSIM: ±{metrics.get('ssim_std', 0):.4f}\n"
        report += f"  MSE: ±{metrics.get('mse_std', 0):.4f}\n"
        report += f"  Color Accuracy: ±{metrics.get('color_accuracy_std', 0):.2f}\n"
        report += f"  Perceptual Loss: ±{metrics.get('perceptual_loss_std', 0):.4f}\n\n"
    
    # Interpretation
    report += "Interpretation:\n"
    report += "- PSNR > 30 dB: Excellent quality\n"
    report += "- SSIM > 0.9: Very high similarity\n"
    report += "- Lower MSE and Color Accuracy values indicate better performance\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report
