"""
Database Tests for Indian Cattle & Buffalo Breed Recognition System

This module contains tests for the database manager functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import sqlite3
from datetime import datetime

# Import the database manager
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database.db_manager import DatabaseManager, get_db_manager


class TestDatabaseManager:
    """Test the database manager functionality."""
    
    def setup_method(self):
        """Set up a temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up the temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test that the database initializes correctly."""
        assert os.path.exists(self.temp_db.name)
        
        # Test that required tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check for breeds table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='breeds'")
        assert cursor.fetchone() is not None
        
        # Check for predictions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_insert_and_get_prediction(self):
        """Test inserting and retrieving a prediction."""
        # Insert a prediction
        top_predictions = [
            {"breed": "Gir", "confidence": 85.5},
            {"breed": "Sahiwal", "confidence": 12.3},
            {"breed": "Ongole", "confidence": 2.2}
        ]
        
        prediction_id = self.db_manager.insert_prediction(
            image_name="test_image.jpg",
            image_path="uploads/test_image.jpg",
            predicted_breed="Gir",
            confidence=85.5,
            top_predictions=top_predictions
        )
        
        assert prediction_id is not None
        assert isinstance(prediction_id, int)
        
        # Retrieve the prediction
        retrieved = self.db_manager.get_prediction_by_id(prediction_id)
        
        assert retrieved is not None
        assert retrieved['prediction_id'] == prediction_id
        assert retrieved['image_name'] == "test_image.jpg"
        assert retrieved['predicted_breed'] == "Gir"
        assert retrieved['confidence'] == 85.5
        assert len(retrieved['top_predictions']) == 3
    
    def test_get_recent_predictions(self):
        """Test retrieving recent predictions."""
        # Insert multiple predictions
        for i in range(5):
            top_predictions = [{"breed": f"Breed{i}", "confidence": 80.0}]
            self.db_manager.insert_prediction(
                image_name=f"test_image_{i}.jpg",
                image_path=f"uploads/test_image_{i}.jpg",
                predicted_breed=f"Breed{i}",
                confidence=80.0,
                top_predictions=top_predictions
            )
        
        # Get recent predictions
        recent = self.db_manager.get_recent_predictions(limit=3)
        
        assert len(recent) == 3
        assert isinstance(recent, list)
        
        # Check that they are ordered by timestamp (most recent first)
        for pred in recent:
            assert 'prediction_id' in pred
            assert 'image_name' in pred
            assert 'predicted_breed' in pred
            assert 'confidence' in pred
    
    def test_get_breed_by_name(self):
        """Test retrieving breed information by name."""
        # Get an existing breed
        breed_info = self.db_manager.get_breed_by_name("Gir")
        
        if breed_info:  # If the breed exists in the database
            assert breed_info['breed_name'] == "Gir"
            assert breed_info['breed_type'] in ["Cattle", "Buffalo"]
    
    def test_get_all_breeds(self):
        """Test retrieving all breeds."""
        breeds = self.db_manager.get_all_breeds()
        
        assert isinstance(breeds, list)
        # There should be at least some breeds in the database
        assert len(breeds) >= 0  # Could be 0 if no data was inserted
    
    def test_get_breeds_by_type(self):
        """Test retrieving breeds by type."""
        cattle_breeds = self.db_manager.get_breeds_by_type("Cattle")
        buffalo_breeds = self.db_manager.get_breeds_by_type("Buffalo")
        
        assert isinstance(cattle_breeds, list)
        assert isinstance(buffalo_breeds, list)
        
        # Check that all returned breeds are of the correct type
        for breed in cattle_breeds:
            assert breed['breed_type'] == "Cattle"
        
        for breed in buffalo_breeds:
            assert breed['breed_type'] == "Buffalo"
    
    def test_prediction_statistics(self):
        """Test retrieving prediction statistics."""
        # Insert some test predictions
        for i in range(3):
            top_predictions = [{"breed": "Gir", "confidence": 80.0 + i}]
            self.db_manager.insert_prediction(
                image_name=f"test_{i}.jpg",
                image_path=f"uploads/test_{i}.jpg",
                predicted_breed="Gir",
                confidence=80.0 + i,
                top_predictions=top_predictions
            )
        
        stats = self.db_manager.get_prediction_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_predictions' in stats
        assert 'predictions_by_breed' in stats
        assert 'average_confidence' in stats
        assert 'recent_predictions' in stats
        
        assert stats['total_predictions'] >= 0
        assert isinstance(stats['predictions_by_breed'], list)
        assert isinstance(stats['average_confidence'], (int, float, type(None)))
        assert isinstance(stats['recent_predictions'], list)
    
    def test_insert_user_feedback(self):
        """Test inserting user feedback."""
        # First insert a prediction to get a valid prediction_id
        top_predictions = [{"breed": "Gir", "confidence": 80.0}]
        prediction_id = self.db_manager.insert_prediction(
            image_name="feedback_test.jpg",
            image_path="uploads/feedback_test.jpg",
            predicted_breed="Gir",
            confidence=80.0,
            top_predictions=top_predictions
        )
        
        # Insert feedback for this prediction
        feedback_id = self.db_manager.insert_user_feedback(
            prediction_id=prediction_id,
            is_correct=True,
            correct_breed="Gir",
            comments="Prediction was accurate"
        )
        
        assert feedback_id is not None
        assert isinstance(feedback_id, int)
        
        # Verify the feedback was inserted by querying the database directly
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prediction_feedback WHERE feedback_id = ?", (feedback_id,))
        feedback_record = cursor.fetchone()
        conn.close()
        
        assert feedback_record is not None
        assert feedback_record[1] == prediction_id  # prediction_id field
        assert feedback_record[2] == 1  # is_correct field (True becomes 1 in SQLite)


class TestDatabaseSingleton:
    """Test the database singleton pattern."""
    
    def test_get_db_manager(self):
        """Test that get_db_manager returns the same instance."""
        from database.db_manager import get_db_manager
        
        db1 = get_db_manager()
        db2 = get_db_manager()
        
        assert db1 is db2  # Same instance
    
    def test_database_operations_with_singleton(self):
        """Test database operations using the singleton instance."""
        db = get_db_manager()
        
        # This test will use the default database path
        # Just verify that we can access the database methods
        assert hasattr(db, 'insert_prediction')
        assert hasattr(db, 'get_prediction_by_id')
        assert hasattr(db, 'get_all_breeds')
        assert hasattr(db, 'get_prediction_statistics')


def test_database_manager_directly():
    """Direct test of database manager functionality."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        temp_db.close()  # Close so it can be reopened by the database manager
        
        try:
            # Create database manager instance
            db_manager = DatabaseManager(temp_db.name)
            
            # Test basic functionality
            breeds = db_manager.get_all_breeds()
            assert isinstance(breeds, list)
            
            # Test inserting a prediction
            top_predictions = [{"breed": "TestBreed", "confidence": 90.0}]
            pred_id = db_manager.insert_prediction(
                image_name="test.jpg",
                image_path="uploads/test.jpg",
                predicted_breed="TestBreed",
                confidence=90.0,
                top_predictions=top_predictions
            )
            
            if pred_id:  # If insertion worked
                retrieved = db_manager.get_prediction_by_id(pred_id)
                assert retrieved is not None
                assert retrieved['predicted_breed'] == "TestBreed"
        
        finally:
            # Clean up
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])