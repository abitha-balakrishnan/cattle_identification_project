"""
Configuration for pytest in the Cattle Breed Recognition System.

This file sets up test configuration and fixtures for the entire test suite.
"""

import pytest
import sys
import os
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up the test environment before running tests.
    """
    # Set environment variables for testing
    os.environ['TESTING'] = 'True'
    
    # Patch any external dependencies that shouldn't be called during tests
    with patch('torch.cuda.is_available', return_value=False):  # Force CPU for tests
        yield
        
        # Cleanup after tests
        if 'TESTING' in os.environ:
            del os.environ['TESTING']


@pytest.fixture
def sample_image():
    """
    Provide a sample image for testing.
    """
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes


@pytest.fixture
def mock_model():
    """
    Provide a mock model for testing.
    """
    from unittest.mock import MagicMock
    import torch
    
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 5)  # Mock output for 5 classes
    mock_model.eval = MagicMock()
    mock_model.to = lambda device: mock_model
    
    return mock_model


@pytest.fixture
def temp_db():
    """
    Provide a temporary database for testing.
    """
    import tempfile
    import os
    from database.db_manager import DatabaseManager
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_file.close()
    
    db_manager = DatabaseManager(temp_file.name)
    
    yield db_manager
    
    # Cleanup
    os.unlink(temp_file.name)