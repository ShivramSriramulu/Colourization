# Image Colorization API Documentation

This document provides comprehensive API documentation for the Image Colorization project.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Data Processing](#data-processing)
3. [Training](#training)
4. [Inference](#inference)
5. [Configuration](#configuration)
6. [Utilities](#utilities)

## Model Architecture

### UnetGenerator

The main generator network based on U-Net architecture with ResNet18 backbone.

```python
class UnetGenerator(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        """
        Initialize U-Net generator
        
        Args:
            input_c (int): Number of input channels (default: 1 for L channel)
            output_c (int): Number of output channels (default: 2 for ab channels)
            n_down (int): Number of downsampling layers (default: 8)
            num_filters (int): Number of filters in first layer (default: 64)
        """
```

**Methods:**
- `forward(input)`: Forward pass through the network

**Example:**
```python
generator = UnetGenerator(input_c=1, output_c=2)
l_channel = torch.randn(1, 1, 256, 256)  # Grayscale input
ab_channels = generator(l_channel)  # Color channels output
```

### PatchDiscriminator

Patch-based discriminator for GAN training.

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        """
        Initialize patch discriminator
        
        Args:
            input_c (int): Number of input channels (L + ab = 3)
            num_filters (int): Number of filters in first layer (default: 64)
            n_down (int): Number of downsampling layers (default: 3)
        """
```

**Methods:**
- `forward(x)`: Forward pass through the discriminator

**Example:**
```python
discriminator = PatchDiscriminator(input_c=3)
combined_input = torch.randn(1, 3, 256, 256)  # L + ab channels
disc_output = discriminator(combined_input)
```

## Data Processing

### ImageDataset

PyTorch dataset for loading and preprocessing images.

```python
class ImageDataset(Dataset):
    def __init__(self, paths, train=True):
        """
        Initialize dataset
        
        Args:
            paths (list): List of image file paths
            train (bool): Whether this is training data (affects augmentation)
        """
```

**Methods:**
- `__len__()`: Return number of images
- `__getitem__(idx)`: Get a single sample

**Returns:**
Dictionary with keys:
- `'L'`: Luminance channel (normalized to [-1, 1])
- `'ab'`: Chrominance channels (normalized to [-1, 1])

**Example:**
```python
dataset = ImageDataset(image_paths, train=True)
sample = dataset[0]
l_channel = sample['L']  # Shape: (1, 256, 256)
ab_channels = sample['ab']  # Shape: (2, 256, 256)
```

## Training

### Main Training Function

```python
def train_model(args):
    """
    Train the colorization model
    
    Args:
        args: ArgumentParser object with training parameters
    """
```

**Training Parameters:**
- `--data_dir`: Directory containing training images
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 2e-4)
- `--img_size`: Image size (default: 256)

**Example:**
```bash
python main.py --mode train --data_dir ./data --epochs 20 --batch_size 32
```

## Inference

### Colorize Single Image

```python
def colorize_image(image_path: str, gen: UnetGenerator, output_path: Optional[str] = None) -> Image.Image:
    """
    Colorize a single grayscale image
    
    Args:
        image_path (str): Path to input image
        gen (UnetGenerator): Trained generator model
        output_path (str, optional): Path to save output image
        
    Returns:
        PIL.Image: Colorized image
    """
```

**Example:**
```python
from main import load_model, colorize_image

# Load trained model
gen, disc = load_model('main-model.pt')

# Colorize image
result = colorize_image('input.jpg', gen, 'output.jpg')
```

### Load Model

```python
def load_model(model_path: str) -> Tuple[UnetGenerator, PatchDiscriminator]:
    """
    Load trained models from checkpoint
    
    Args:
        model_path (str): Path to model checkpoint
        
    Returns:
        tuple: (generator, discriminator) models
    """
```

## Configuration

### Config Class

Centralized configuration management.

```python
class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.output = OutputConfig()
```

**Configuration Sections:**

#### ModelConfig
- `input_channels`: Number of input channels (default: 1)
- `output_channels`: Number of output channels (default: 2)
- `num_filters`: Number of filters in first layer (default: 64)
- `n_down`: Number of downsampling layers (default: 8)
- `dropout_rate`: Dropout rate (default: 0.5)

#### TrainingConfig
- `image_size`: Image dimensions (default: (256, 256))
- `batch_size`: Batch size (default: 32)
- `epochs`: Number of epochs (default: 20)
- `generator_lr`: Generator learning rate (default: 2e-4)
- `discriminator_lr`: Discriminator learning rate (default: 2e-4)
- `lambda_l1`: L1 loss weight (default: 100.0)

#### DataConfig
- `train_size`: Number of training images (default: 8000)
- `val_size`: Number of validation images (default: 2000)
- `data_dir`: Data directory path (default: "./data")
- `image_extensions`: Supported image formats

**Example:**
```python
from config import Config

config = Config()
config.training.batch_size = 64
config.model.num_filters = 128
config.create_directories()
```

## Utilities

### Image Processing

```python
def preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Preprocess image for model input"""

def rgb_to_lab(rgb_image: torch.Tensor) -> torch.Tensor:
    """Convert RGB image to LAB color space"""

def lab_to_rgb(lab_image: torch.Tensor) -> np.ndarray:
    """Convert LAB image to RGB color space"""
```

### Evaluation Metrics

```python
def calculate_metrics(original: np.ndarray, colorized: np.ndarray) -> dict:
    """
    Calculate image quality metrics
    
    Returns:
        dict: Dictionary with 'ssim' and 'psnr' scores
    """
```

### Visualization

```python
def visualize_results(original, colorized, grayscale, save_path=None):
    """Visualize original, grayscale, and colorized images side by side"""

def create_training_plots(losses: dict, save_dir: str):
    """Create and save training loss plots"""
```

### Data Loading

```python
def load_image_paths(data_dir: str, extensions: List[str] = ['*.jpg', '*.jpeg', '*.png']) -> List[str]:
    """Load image paths from directory"""

def validate_image_paths(image_paths: List[str]) -> List[str]:
    """Validate that all image paths exist and are readable"""
```

## Command Line Interface

### Training Mode
```bash
python main.py --mode train \
    --data_dir /path/to/images \
    --epochs 20 \
    --batch_size 32 \
    --lr 2e-4 \
    --img_size 256 \
    --output_dir ./outputs
```

### Evaluation Mode
```bash
python main.py --mode eval \
    --model_path main-model.pt \
    --test_dir test_images/ \
    --output_dir ./results
```

### Inference Mode
```bash
python main.py --mode inference \
    --model_path main-model.pt \
    --input_image test1.jpg \
    --output_dir ./outputs
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid image files
- Missing model checkpoints
- Incorrect input dimensions
- Memory issues during training
- Invalid configuration parameters

## Performance Considerations

- **Memory**: Training requires ~4GB GPU memory for batch_size=32
- **Speed**: Inference takes ~0.1 seconds per image on GPU
- **Storage**: Model checkpoints are ~119MB each
- **Scalability**: Supports batch processing for multiple images

## Best Practices

1. **Data Preparation**: Ensure images are in RGB format
2. **Model Loading**: Always use `load_model()` function
3. **Configuration**: Validate config before training
4. **Memory Management**: Monitor GPU memory during training
5. **Checkpointing**: Save models regularly during training
