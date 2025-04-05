"""
Unit tests for Image Colorization model components

This module contains comprehensive tests for the model architecture,
data loading, and utility functions.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from main import UnetGenerator, PatchDiscriminator, ImageDataset, init_weights
from utils import preprocess_image, rgb_to_lab, lab_to_rgb, calculate_metrics
from config import Config

class TestModelArchitecture(unittest.TestCase):
    """Test model architecture components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.image_size = 256
        self.config = Config()
    
    def test_generator_architecture(self):
        """Test U-Net generator architecture"""
        generator = UnetGenerator(
            input_c=1,
            output_c=2,
            n_down=8,
            num_filters=64
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, 1, self.image_size, self.image_size)
        output = generator(x)
        
        # Check output shape
        expected_shape = (self.batch_size, 2, self.image_size, self.image_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output range (should be between -1 and 1 due to Tanh)
        self.assertTrue(torch.all(output >= -1))
        self.assertTrue(torch.all(output <= 1))
    
    def test_discriminator_architecture(self):
        """Test patch discriminator architecture"""
        discriminator = PatchDiscriminator(
            input_c=3,
            num_filters=64,
            n_down=3
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        output = discriminator(x)
        
        # Check output shape (should be patch-based)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 1)
        
        # Check that output is smaller than input (due to downsampling)
        self.assertLess(output.shape[2], self.image_size)
        self.assertLess(output.shape[3], self.image_size)
    
    def test_weight_initialization(self):
        """Test weight initialization"""
        generator = UnetGenerator()
        discriminator = PatchDiscriminator(input_c=3)
        
        # Initialize weights
        init_weights(generator)
        init_weights(discriminator)
        
        # Check that weights are not all zero
        for name, param in generator.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))
        
        for name, param in discriminator.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))

class TestDataProcessing(unittest.TestCase):
    """Test data processing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image_size = (256, 256)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = os.path.join(self.temp_dir, 'test.jpg')
        
        from PIL import Image
        Image.fromarray(dummy_image).save(image_path)
        
        # Test preprocessing
        processed = preprocess_image(image_path, self.test_image_size)
        
        # Check output type and shape
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (3, self.test_image_size[0], self.test_image_size[1]))
        
        # Check value range (should be between 0 and 1)
        self.assertTrue(torch.all(processed >= 0))
        self.assertTrue(torch.all(processed <= 1))
    
    def test_rgb_to_lab_conversion(self):
        """Test RGB to LAB color space conversion"""
        # Create dummy RGB image
        rgb_image = torch.randn(3, 64, 64)
        
        # Convert to LAB
        lab_image = rgb_to_lab(rgb_image)
        
        # Check output shape
        self.assertEqual(lab_image.shape, (3, 64, 64))
        
        # Check that L channel is in expected range
        l_channel = lab_image[0]
        self.assertTrue(torch.all(l_channel >= 0))
        self.assertTrue(torch.all(l_channel <= 100))
    
    def test_lab_to_rgb_conversion(self):
        """Test LAB to RGB color space conversion"""
        # Create dummy LAB image
        lab_image = torch.randn(3, 64, 64)
        lab_image[0] = lab_image[0] * 50 + 50  # L channel: 0-100
        lab_image[1:] = lab_image[1:] * 110  # ab channels: -110 to 110
        
        # Convert to RGB
        rgb_image = lab_to_rgb(lab_image)
        
        # Check output type and shape
        self.assertIsInstance(rgb_image, np.ndarray)
        self.assertEqual(rgb_image.shape, (64, 64, 3))
        
        # Check value range (should be between 0 and 255)
        self.assertTrue(np.all(rgb_image >= 0))
        self.assertTrue(np.all(rgb_image <= 255))

class TestMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        # Create dummy images
        original = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        colorized = original + np.random.randint(-10, 10, (64, 64, 3), dtype=np.uint8)
        colorized = np.clip(colorized, 0, 255)
        
        # Calculate metrics
        metrics = calculate_metrics(original, colorized)
        
        # Check that metrics are calculated
        self.assertIn('ssim', metrics)
        self.assertIn('psnr', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['ssim'], 0)
        self.assertLessEqual(metrics['ssim'], 1)
        self.assertGreaterEqual(metrics['psnr'], 0)

class TestDataset(unittest.TestCase):
    """Test dataset functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_dummy_images()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_dummy_images(self):
        """Create dummy images for testing"""
        from PIL import Image
        
        for i in range(5):
            # Create random image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(self.temp_dir, f'test_{i}.jpg'))
    
    def test_dataset_creation(self):
        """Test dataset creation and loading"""
        image_paths = [os.path.join(self.temp_dir, f'test_{i}.jpg') for i in range(5)]
        
        # Create dataset
        dataset = ImageDataset(image_paths, train=True)
        
        # Check dataset length
        self.assertEqual(len(dataset), 5)
        
        # Test getting an item
        sample = dataset[0]
        
        # Check sample structure
        self.assertIn('L', sample)
        self.assertIn('ab', sample)
        
        # Check shapes
        self.assertEqual(sample['L'].shape, (1, 256, 256))
        self.assertEqual(sample['ab'].shape, (2, 256, 256))

class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_creation(self):
        """Test configuration object creation"""
        config = Config()
        
        # Check that all config sections exist
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.logging)
        self.assertIsNotNone(config.output)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        
        # Test valid configuration
        self.assertTrue(config.validate())
        
        # Test invalid learning rate
        config.training.generator_lr = -1
        self.assertFalse(config.validate())
        
        # Reset to valid value
        config.training.generator_lr = 2e-4
        self.assertTrue(config.validate())
    
    def test_directory_creation(self):
        """Test directory creation"""
        config = Config()
        
        # Change paths to temp directory
        temp_dir = tempfile.mkdtemp()
        config.data.data_dir = os.path.join(temp_dir, "data")
        config.output.output_dir = os.path.join(temp_dir, "outputs")
        
        # Create directories
        config.create_directories()
        
        # Check that directories were created
        self.assertTrue(os.path.exists(config.data.data_dir))
        self.assertTrue(os.path.exists(config.output.output_dir))
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
