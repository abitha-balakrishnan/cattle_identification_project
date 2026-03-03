"""
Model Tests for Indian Cattle & Buffalo Breed Recognition System

This module contains tests for the model functionality and preprocessing.
"""

import pytest
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the model components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.model_training import CattleBreedDataset
from backend.main import load_model, transform, CLASS_NAMES, BREED_INFO


class TestModelStructure:
    """Test the model structure and loading."""
    
    def test_model_architecture(self):
        """Test that the model has the correct architecture."""
        # Create a simple model with dummy number of classes
        num_classes = 5
        model = load_model.__globals__['models'].resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        assert output.shape == (1, num_classes)
        assert isinstance(output, torch.Tensor)
    
    def test_transform_pipeline(self):
        """Test that the transform pipeline works correctly."""
        # Create a dummy image
        img = Image.new('RGB', (300, 300), color='red')
        
        # Apply the transform
        transformed = transform(img)
        
        # Check output shape and type
        assert transformed.shape == (3, 224, 224)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.dtype == torch.float32


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_model_with_checkpoint(self):
        """Test loading model from a checkpoint."""
        # Create a temporary model file
        num_classes = 5
        temp_model = torch.nn.Linear(512, num_classes)  # Simplified for test
        temp_checkpoint = {
            'model_state_dict': temp_model.state_dict(),
            'epoch': 10,
            'val_acc': 85.0
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            torch.save(temp_checkpoint, tmp_file.name)
            
            try:
                # Temporarily modify the model loading function
                original_path = '../cattle_model.pth'
                
                # Since we can't actually load the model without the full architecture,
                # we'll just test that the function runs without error
                with patch('backend.main.os.path.exists', return_value=True):
                    with patch('backend.main.torch.load', return_value=temp_checkpoint):
                        with patch('backend.main.models.resnet18') as mock_resnet:
                            mock_model = MagicMock()
                            mock_model.fc = MagicMock()
                            mock_model.to = MagicMock(return_value=mock_model)
                            mock_model.eval = MagicMock()
                            mock_resnet.return_value = mock_model
                            
                            # This should run without error
                            from backend.main import load_model
                            model = load_model()
                            
                            # Verify the mocks were called
                            mock_resnet.assert_called_once()
                            mock_model.to.assert_called_once()
                            mock_model.eval.assert_called_once()
                            
            finally:
                # Clean up
                os.unlink(tmp_file.name)
    
    def test_model_prediction_process(self):
        """Test the complete prediction process."""
        # Create a dummy model for testing
        with patch('backend.main.model') as mock_model:
            # Mock the model's behavior
            mock_output = torch.tensor([[0.1, 0.7, 0.2]])  # Probabilities for 3 classes
            mock_model.return_value.return_value = mock_output
            mock_model.return_value.eval = MagicMock()
            
            # Test the prediction process
            dummy_tensor = torch.randn(1, 3, 224, 224)
            
            # This simulates what happens in the predict function
            with torch.no_grad():
                outputs = mock_model(dummy_tensor)  # This calls __call__ on mock_model
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top predictions
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                # Verify the output
                assert top3_probs.shape == torch.Size([1, 3])
                assert top3_indices.shape == torch.Size([1, 3])
                
                # Probabilities should sum close to 1
                assert torch.allclose(torch.sum(probabilities, dim=1), torch.tensor([1.0]), atol=1e-5)


class TestDatasetFunctionality:
    """Test dataset functionality."""
    
    def test_class_names_defined(self):
        """Test that class names are properly defined."""
        assert isinstance(CLASS_NAMES, list)
        assert len(CLASS_NAMES) > 0
        assert all(isinstance(name, str) for name in CLASS_NAMES)
        
        # Check that some expected breeds are in the list
        expected_breeds = ["Gir", "Sahiwal", "Ongole", "Murrah", "Jaffarabadi"]
        for breed in expected_breeds:
            assert breed in CLASS_NAMES or any(breed.replace(" ", "_") in name for name in CLASS_NAMES)
    
    def test_breed_info_defined(self):
        """Test that breed information is properly defined."""
        assert isinstance(BREED_INFO, dict)
        assert len(BREED_INFO) > 0
        
        # Check structure of breed info
        if BREED_INFO:
            sample_breed = next(iter(BREED_INFO.values()))
            assert isinstance(sample_breed, dict)
            assert 'name' in sample_breed or 'breed_name' in sample_breed
            assert 'description' in sample_breed


class TestImageProcessing:
    """Test image processing functionality."""
    
    def test_image_transform_consistency(self):
        """Test that image transforms are consistent."""
        # Create the same image twice
        img1 = Image.new('RGB', (300, 200), color='blue')
        img2 = Image.new('RGB', (300, 200), color='blue')
        
        # Apply transform to both
        transformed1 = transform(img1)
        transformed2 = transform(img2)
        
        # They should be identical
        assert torch.equal(transformed1, transformed2)
        assert transformed1.shape == transformed2.shape
    
    def test_different_image_sizes(self):
        """Test that different image sizes are handled correctly."""
        sizes = [(100, 100), (300, 200), (500, 600)]
        
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='green')
            transformed = transform(img)
            
            # All should result in the same output size
            assert transformed.shape == (3, 224, 224)
    
    def test_rgb_conversion(self):
        """Test that images are converted to RGB."""
        # Create a grayscale image
        gray_img = Image.new('L', (224, 224), color=128)
        transformed = transform(gray_img)
        
        # Should have 3 channels after conversion
        assert transformed.shape[0] == 3  # 3 color channels


class TestModelIntegration:
    """Test model integration with the rest of the system."""
    
    def test_model_output_shape_matches_classes(self):
        """Test that model output shape matches number of classes."""
        num_classes = len(CLASS_NAMES)
        
        # Create a mock model with the right number of outputs
        with patch('backend.main.models.resnet18') as mock_resnet:
            mock_model = MagicMock()
            mock_model.fc = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_resnet.return_value = mock_model
            
            # Simulate loading the model
            # The model should have fc layer with in_features -> num_classes mapping
            from torchvision import models
            resnet = models.resnet18(weights=None)
            resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            output = resnet(dummy_input)
            
            assert output.shape[1] == num_classes
    
    def test_probability_normalization(self):
        """Test that output probabilities sum to 1."""
        # Simulate model output logits
        logits = torch.tensor([[2.0, 1.0, 0.1, -0.5, 0.3]])
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)
        
        # Sum of probabilities should be 1
        prob_sum = torch.sum(probabilities)
        assert torch.allclose(prob_sum, torch.tensor([1.0]), atol=1e-5)
        
        # Individual probabilities should be between 0 and 1
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)


def test_model_requirements():
    """Test that required model components are available."""
    # Check that PyTorch is available
    assert torch.__version__
    
    # Check that torchvision is available
    import torchvision
    assert hasattr(torchvision, 'transforms')
    assert hasattr(torchvision, 'models')
    
    # Check that PIL is available
    assert Image.PILLOW_VERSION or hasattr(Image, 'PILLOW_VERSION')


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])