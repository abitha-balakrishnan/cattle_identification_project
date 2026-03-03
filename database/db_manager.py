"""
Database Manager for Indian Cattle & Buffalo Breed Recognition System

This module handles all database operations for the cattle breed recognition system
using SQLite and SQLAlchemy ORM.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for the cattle breed recognition system."""
    
    def __init__(self, db_path: str = "./cattle_recognition.db"):
        """Initialize the database manager."""
        self.db_path = Path(db_path)
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables and sample data."""
        # Create the database file if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Read and execute schema SQL
        schema_file = Path(__file__).parent / "db_schema.sql"
        if schema_file.exists():
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            cursor.executescript(schema_sql)
            logger.info(f"Database schema initialized at {self.db_path}")
        else:
            logger.error(f"Schema file not found: {schema_file}")
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def insert_prediction(self, image_name: str, image_path: str, predicted_breed: str, 
                        confidence: float, top_predictions: List[Dict[str, Any]], 
                        user_ip: str = None, user_agent: str = None) -> int:
        """Insert a prediction record into the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            top_predictions_json = json.dumps(top_predictions)
            
            cursor.execute("""
                INSERT INTO predictions 
                (image_name, image_path, predicted_breed, confidence, top_predictions, user_ip, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_name, image_path, predicted_breed, confidence, top_predictions_json, user_ip, user_agent))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Prediction inserted with ID: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting prediction: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a prediction by its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT prediction_id, image_name, image_path, predicted_breed, 
                       confidence, top_predictions, timestamp
                FROM predictions 
                WHERE prediction_id = ?
            """, (prediction_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'prediction_id': row[0],
                    'image_name': row[1],
                    'image_path': row[2],
                    'predicted_breed': row[3],
                    'confidence': row[4],
                    'top_predictions': json.loads(row[5]) if row[5] else [],
                    'timestamp': row[6]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving prediction: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent prediction records."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT prediction_id, image_name, image_path, predicted_breed, 
                       confidence, timestamp
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                predictions.append({
                    'prediction_id': row[0],
                    'image_name': row[1],
                    'image_path': row[2],
                    'predicted_breed': row[3],
                    'confidence': row[4],
                    'timestamp': row[5]
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving recent predictions: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_breed_by_name(self, breed_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve breed information by name."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM breeds WHERE breed_name = ?", (breed_name,))
            row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                breed_info = dict(zip(columns, row))
                return breed_info
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving breed: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_all_breeds(self) -> List[Dict[str, Any]]:
        """Retrieve all breed information."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM breeds ORDER BY breed_name")
            rows = cursor.fetchall()
            
            breeds = []
            columns = [description[0] for description in cursor.description]
            
            for row in rows:
                breed_info = dict(zip(columns, row))
                breeds.append(breed_info)
            
            return breeds
            
        except Exception as e:
            logger.error(f"Error retrieving all breeds: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_breeds_by_type(self, breed_type: str) -> List[Dict[str, Any]]:
        """Retrieve breeds filtered by type (Cattle or Buffalo)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM breeds WHERE breed_type = ? ORDER BY breed_name", (breed_type,))
            rows = cursor.fetchall()
            
            breeds = []
            columns = [description[0] for description in cursor.description]
            
            for row in rows:
                breed_info = dict(zip(columns, row))
                breeds.append(breed_info)
            
            return breeds
            
        except Exception as e:
            logger.error(f"Error retrieving breeds by type: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Predictions by breed
            cursor.execute("""
                SELECT predicted_breed, COUNT(*) as count 
                FROM predictions 
                GROUP BY predicted_breed 
                ORDER BY count DESC
            """)
            breed_counts = cursor.fetchall()
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM predictions")
            avg_confidence = cursor.fetchone()[0]
            
            # Recent predictions
            cursor.execute("""
                SELECT predicted_breed, confidence, timestamp 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_predictions = cursor.fetchall()
            
            stats = {
                'total_predictions': total_predictions,
                'predictions_by_breed': [{'breed': bc[0], 'count': bc[1]} for bc in breed_counts],
                'average_confidence': round(avg_confidence, 2) if avg_confidence else 0,
                'recent_predictions': [
                    {'breed': rp[0], 'confidence': rp[1], 'timestamp': rp[2]}
                    for rp in recent_predictions
                ]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting prediction statistics: {str(e)}")
            return {}
        finally:
            conn.close()
    
    def insert_user_feedback(self, prediction_id: int, is_correct: bool, 
                           correct_breed: str = None, comments: str = None) -> int:
        """Insert user feedback for a prediction."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO prediction_feedback 
                (prediction_id, is_correct, correct_breed, comments)
                VALUES (?, ?, ?, ?)
            """, (prediction_id, is_correct, correct_breed, comments))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Feedback inserted with ID: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting feedback: {str(e)}")
            raise
        finally:
            conn.close()


# Singleton instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the database manager instance."""
    return db_manager


if __name__ == "__main__":
    # Test the database manager
    print("Testing Database Manager...")
    
    # Initialize database
    db = get_db_manager()
    
    # Test breed retrieval
    print("\nAll Breeds:")
    breeds = db.get_all_breeds()
    for breed in breeds[:5]:  # Show first 5 breeds
        print(f"  {breed['breed_name']} ({breed['breed_type']}) - {breed['origin']}")
    
    print(f"\nTotal breeds: {len(breeds)}")
    
    # Test breed by type
    cattle_breeds = db.get_breeds_by_type('Cattle')
    buffalo_breeds = db.get_breeds_by_type('Buffalo')
    
    print(f"Cattle breeds: {len(cattle_breeds)}")
    print(f"Buffalo breeds: {len(buffalo_breeds)}")
    
    print("\nDatabase manager test completed successfully!")