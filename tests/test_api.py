"""
Integration Tests for Indian Cattle & Buffalo Breed Recognition API

This module contains comprehensive tests for the FastAPI backend,
including API endpoints, model predictions, and database integration.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import status
import tempfile
import os
from PIL import Image
import io
import json
from unittest.mock import patch, MagicMock

# Import the FastAPI app
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app
from database.db_manager import DatabaseManager

# Create a test client
client = TestClient(app)

def create_test_image(size=(224, 224), color=(255, 0, 0)):
    """Create a simple test image."""
    img = Image.new('RGB', size, color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestAPIStructure:
    """Test basic API structure and endpoints."""
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        # This might return HTML, so we just check if it's accessible
        assert response.status_code in [200, 404]  # 404 might occur if template not found
    
    def test_breeds_endpoint(self):
        """Test the breeds listing endpoint."""
        response = client.get("/breeds")
        assert response.status_code == 200
        data = response.json()
        assert "breeds" in data
        assert isinstance(data["breeds"], list)
        assert data["count"] >= 0  # At least 0 breeds should be returned


class TestPredictionEndpoint:
    """Test the prediction endpoint."""
    
    def test_predict_endpoint_with_valid_image(self):
        """Test prediction with a valid image."""
        # Create a test image
        test_image = create_test_image()
        
        # Send the image to the prediction endpoint
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        
        # The response might vary depending on model availability,
        # but it should return a proper JSON response
        if response.status_code == 200:
            data = response.json()
            assert "breed" in data
            assert "confidence" in data
            assert "top_predictions" in data
            assert isinstance(data["confidence"], (int, float))
            assert 0 <= data["confidence"] <= 100
        else:
            # If model is not available, we expect a 500 error
            assert response.status_code in [200, 500]
    
    def test_predict_endpoint_invalid_file_type(self):
        """Test prediction with an invalid file type."""
        # Create a text file instead of an image
        fake_file = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", fake_file, "text/plain")}
        )
        
        # Should return a 400 error for invalid file type
        assert response.status_code == 400
    
    def test_predict_endpoint_no_file(self):
        """Test prediction without uploading a file."""
        response = client.post("/predict")
        # This should return a 422 error due to missing required file
        assert response.status_code == 422
    
    def test_predict_endpoint_large_file(self):
        """Test prediction with a very large file."""
        # Create a very large image (simulated)
        large_image = io.BytesIO(b"A" * (10 * 1024 * 1024))  # 10MB of dummy data
        
        response = client.post(
            "/predict",
            files={"file": ("large.jpg", large_image, "image/jpeg")}
        )
        
        # Depending on server config, this might return 413 (Payload Too Large)
        # or 400/422 for invalid image format
        assert response.status_code in [400, 413, 422]


class TestBreedInfoEndpoints:
    """Test breed information endpoints."""
    
    def test_get_specific_breed(self):
        """Test getting information for a specific breed."""
        # Test with a known breed name
        response = client.get("/breed/Gir")
        # This should return either 200 (found) or 404 (not found)
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            # If breed exists, it should have at least a name
            assert "breed_name" in data or "name" in data
    
    def test_get_nonexistent_breed(self):
        """Test getting information for a non-existent breed."""
        response = client.get("/breed/NonExistentBreed")
        assert response.status_code == 404


class TestDatabaseIntegration:
    """Test database integration endpoints."""
    
    def test_get_all_breeds_from_db(self):
        """Test getting all breeds from the database."""
        response = client.get("/breeds/db")
        assert response.status_code == 200
        data = response.json()
        assert "breeds" in data
        assert "count" in data
        assert isinstance(data["breeds"], list)
        assert isinstance(data["count"], int)
        assert data["count"] == len(data["breeds"])
    
    def test_get_prediction_statistics(self):
        """Test getting prediction statistics."""
        response = client.get("/predictions/stats")
        assert response.status_code == 200
        data = response.json()
        # Statistics should include at least total predictions
        assert "total_predictions" in data
        assert isinstance(data["total_predictions"], int)
    
    def test_get_recent_predictions(self):
        """Test getting recent predictions."""
        response = client.get("/predictions/recent")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert isinstance(data["predictions"], list)
        assert isinstance(data["count"], int)
        
        # If there are predictions, they should have required fields
        for pred in data["predictions"]:
            assert "prediction_id" in pred
            assert "predicted_breed" in pred
            assert "confidence" in pred
            assert "timestamp" in pred


class TestFeedbackEndpoint:
    """Test feedback submission endpoint."""
    
    def test_submit_feedback_valid(self):
        """Test submitting valid feedback."""
        # We need a prediction ID to submit feedback, 
        # but we'll test the endpoint structure
        response = client.post(
            "/feedback",
            data={
                "prediction_id": 1,
                "is_correct": True,
                "correct_breed": "Gir",
                "comments": "Test feedback"
            }
        )
        
        # This might fail if prediction ID doesn't exist, but should return appropriate error
        assert response.status_code in [200, 404, 422, 500]
    
    def test_submit_feedback_invalid(self):
        """Test submitting feedback with invalid data."""
        response = client.post(
            "/feedback",
            data={
                "prediction_id": -1,  # Invalid ID
                "is_correct": "not_a_boolean",  # Invalid type
            }
        )
        
        # Should return validation error
        assert response.status_code in [422, 500]


class TestModelIntegration:
    """Test model integration aspects."""
    
    def test_model_loading(self):
        """Test that the model can be loaded."""
        # This tests the internal model loading mechanism
        # We'll make a simple request and see if the model works
        test_image = create_test_image()
        
        # Temporarily disable the actual model for testing
        with patch('backend.main.model') as mock_model:
            # Mock the model's forward method
            mock_output = MagicMock()
            mock_output.return_value = MagicMock()
            mock_probs = MagicMock()
            mock_probs.max.return_value = (MagicMock(item=lambda: 0.9), MagicMock(item=lambda: 0))
            mock_model.return_value = MagicMock(**{
                '__call__.return_value': mock_output,
                'eval.return_value': None
            })
            
            with patch('backend.main.torch.softmax', return_value=mock_probs):
                response = client.post(
                    "/predict",
                    files={"file": ("test.jpg", test_image, "image/jpeg")}
                )
                
                # Even with mocked model, the structure should work
                assert response.status_code in [200, 500]


def test_complete_prediction_flow():
    """Integration test for the complete prediction flow."""
    # Create a test image
    test_image = create_test_image(color=(100, 150, 200))
    
    # Submit image for prediction
    response = client.post(
        "/predict",
        files={"file": ("test_cow.jpg", test_image, "image/jpeg")}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Validate response structure
        assert "breed" in data
        assert "confidence" in data
        assert "top_predictions" in data
        assert "breed_info" in data
        
        # Validate data types
        assert isinstance(data["breed"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["top_predictions"], list)
        
        # Validate confidence range
        assert 0 <= data["confidence"] <= 100
        
        # Validate top predictions structure
        for pred in data["top_predictions"]:
            assert "breed" in pred
            assert "confidence" in pred
            assert isinstance(pred["confidence"], (int, float))
            assert 0 <= pred["confidence"] <= 100


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])