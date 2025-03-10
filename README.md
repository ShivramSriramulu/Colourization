# Image Colorization with GAN

A deep learning project that automatically colorizes grayscale images using a Generative Adversarial Network (GAN) architecture. The model uses a U-Net generator with ResNet18 backbone and a discriminator to produce realistic colorized images from black and white inputs.

## Features

- **GAN-based Architecture**: Uses generator-discriminator framework for high-quality colorization
- **U-Net Generator**: Leverages ResNet18 backbone with skip connections for detailed feature preservation
- **LAB Color Space**: Converts RGB images to LAB color space for better color learning
- **Data Augmentation**: Implements random horizontal flipping for improved generalization
- **Pretrained Models**: Includes multiple trained model checkpoints for immediate inference
- **Interactive Notebooks**: Jupyter notebooks for training, evaluation, and demonstration

## Tech Stack

- **Python**: 3.8+
- **PyTorch**: Deep learning framework
- **FastAI**: High-level API for computer vision
- **OpenCV/PIL**: Image processing
- **NumPy/Matplotlib**: Numerical computing and visualization
- **scikit-image**: Color space conversions

## Repository Structure

```
Colorization/
├── Main.ipynb                 # Main training notebook
├── image_colorization.ipynb   # Colorization implementation
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── scripts/                  # Helper scripts
│   ├── train.sh             # Training script
│   └── eval.sh              # Evaluation script
├── test_images/              # Test images for inference
├── main-model.pt             # Trained model checkpoint
├── model.pt                  # Alternative model checkpoint
└── res18-unet.pt            # ResNet18-U-Net model
```

## Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart

### Training
```bash
# Using the provided script
./scripts/train.sh

# Or directly with main.py
python main.py --data_dir /path/to/images --epochs 20 --batch_size 32
```

### Evaluation
```bash
# Using the provided script
./scripts/eval.sh

# Or directly with main.py
python main.py --mode eval --model_path main-model.pt --test_dir test_images/
```

## Data

The model expects RGB images in JPG format. For training:
- Place your images in a directory structure
- Images will be automatically resized to 256x256
- Data augmentation includes random horizontal flipping

**Data Source**: The original implementation uses COCO dataset samples. For your own data, place images in a directory and update the data path in the training script.

## Training & Evaluation

### Training Command
```bash
python main.py \
    --data_dir /path/to/training/images \
    --epochs 20 \
    --batch_size 32 \
    --lr 2e-4 \
    --img_size 256 \
    --output_dir ./outputs
```

### Evaluation Command
```bash
python main.py \
    --mode eval \
    --model_path main-model.pt \
    --test_dir test_images/ \
    --output_dir ./results
```

## Results

The model achieves realistic colorization results on various image types:
- **Architecture**: GAN with U-Net generator and discriminator
- **Training Time**: ~2-3 hours on GPU for 20 epochs
- **Model Size**: ~119MB (main-model.pt)
- **Input Size**: 256x256 RGB images
- **Output**: Colorized images in LAB color space

## Inference

### Command Line
```bash
python main.py --mode inference --model_path main-model.pt --input_image test1.jpg
```

### Python API
```python
from main import load_model, colorize_image

model = load_model('main-model.pt')
colorized = colorize_image('test1.jpg', model)
colorized.save('colorized_test1.jpg')
```

## Model Card

### Intended Use
- Colorizing historical black and white photographs
- Enhancing grayscale images for artistic purposes
- Research and educational applications in computer vision

### Limitations
- Works best with natural images (landscapes, portraits)
- May struggle with highly abstract or synthetic images
- Requires sufficient training data for optimal results
- Color accuracy depends on training data quality

### Performance
- Training: 20 epochs on 8000 images
- Inference: ~0.1 seconds per image on GPU
- Memory: ~4GB GPU memory required for training

## Roadmap

- [ ] Add support for different image sizes
- [ ] Implement batch processing for multiple images
- [ ] Add color palette customization options
- [ ] Create web interface for easy usage
- [ ] Optimize model for mobile deployment
- [ ] Add support for video colorization
- [ ] Implement attention mechanisms for better detail preservation
- [ ] Create model distillation for faster inference
