"""
Configuration settings for Image Colorization project

This module centralizes all hyperparameters, model configurations, and training settings
for easy modification and experimentation.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Generator settings
    input_channels: int = 1  # L channel
    output_channels: int = 2  # ab channels
    num_filters: int = 64
    n_down: int = 8  # Number of downsampling layers
    
    # Discriminator settings
    disc_input_channels: int = 3  # L + ab channels
    disc_num_filters: int = 64
    disc_n_down: int = 3
    
    # U-Net settings
    dropout_rate: float = 0.5
    leaky_relu_slope: float = 0.2

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data settings
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 32
    num_workers: int = 4
    
    # Training parameters
    epochs: int = 20
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    
    # Data augmentation
    use_horizontal_flip: bool = True
    use_random_crop: bool = False
    use_color_jitter: bool = False
    
    # Training schedule
    pretrain_epochs: int = 5
    pretrain_lr: float = 1e-4
    
    # Checkpointing
    save_frequency: int = 5
    validation_frequency: int = 2

@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset settings
    train_size: int = 8000
    val_size: int = 2000
    external_data_size: int = 10000
    
    # Data paths
    data_dir: str = "./data"
    train_dir: str = "./data/train"
    val_dir: str = "./data/val"
    test_dir: str = "./test_images"
    
    # Supported formats
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Preprocessing
    normalize_l: bool = True
    normalize_ab: bool = True
    l_scale: float = 50.0
    ab_scale: float = 110.0

@dataclass
class LoggingConfig:
    """Logging and visualization configuration"""
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "training.log"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    
    # Visualization
    save_samples: bool = True
    sample_frequency: int = 100
    num_samples: int = 8
    
    # Metrics
    save_metrics: bool = True
    metrics_file: str = "metrics.json"

@dataclass
class OutputConfig:
    """Output configuration"""
    # Directories
    output_dir: str = "./outputs"
    model_dir: str = "./models"
    results_dir: str = "./results"
    
    # File naming
    model_prefix: str = "colorization"
    checkpoint_format: str = "epoch_{epoch:03d}.pt"
    
    # Save options
    save_best_model: bool = True
    save_last_model: bool = True
    save_optimizer: bool = True

class Config:
    """Main configuration class that combines all configs"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.output = OutputConfig()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data.data_dir,
            self.data.train_dir,
            self.data.val_dir,
            self.output.output_dir,
            self.output.model_dir,
            self.output.results_dir,
            self.logging.tensorboard_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_path(self, epoch: Optional[int] = None) -> str:
        """Get model file path"""
        if epoch is not None:
            filename = self.output.checkpoint_format.format(epoch=epoch)
        else:
            filename = f"{self.output.model_prefix}_latest.pt"
        
        return os.path.join(self.output.model_dir, filename)
    
    def get_output_path(self, filename: str) -> str:
        """Get output file path"""
        return os.path.join(self.output.output_dir, filename)
    
    def get_results_path(self, filename: str) -> str:
        """Get results file path"""
        return os.path.join(self.output.results_dir, filename)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        # Check if data directories exist
        if not os.path.exists(self.data.data_dir):
            print(f"Warning: Data directory {self.data.data_dir} does not exist")
        
        # Validate image size
        if self.training.image_size[0] != self.training.image_size[1]:
            print("Warning: Non-square image size may cause issues")
        
        # Validate learning rates
        if self.training.generator_lr <= 0 or self.training.discriminator_lr <= 0:
            print("Error: Learning rates must be positive")
            return False
        
        # Validate batch size
        if self.training.batch_size <= 0:
            print("Error: Batch size must be positive")
            return False
        
        return True

# Default configuration instance
config = Config()

# Environment-specific configurations
def get_production_config():
    """Get production configuration with optimized settings"""
    config = Config()
    config.training.batch_size = 16  # Smaller batch size for stability
    config.training.epochs = 50  # More epochs for better results
    config.logging.use_tensorboard = False  # Disable in production
    return config

def get_debug_config():
    """Get debug configuration with minimal settings"""
    config = Config()
    config.training.batch_size = 4
    config.training.epochs = 2
    config.data.train_size = 100
    config.data.val_size = 20
    config.logging.sample_frequency = 10
    return config

def get_experiment_config(experiment_name: str):
    """Get configuration for specific experiments"""
    config = Config()
    
    if experiment_name == "high_resolution":
        config.training.image_size = (512, 512)
        config.training.batch_size = 8
        config.model.num_filters = 128
    
    elif experiment_name == "fast_training":
        config.training.batch_size = 64
        config.training.generator_lr = 5e-4
        config.training.discriminator_lr = 5e-4
    
    elif experiment_name == "attention":
        # Add attention mechanism configuration
        config.model.use_attention = True
        config.model.attention_heads = 8
    
    return config
