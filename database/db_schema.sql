-- Database Schema for Indian Cattle & Buffalo Breed Recognition System
-- SQLite Database Schema

PRAGMA foreign_keys = ON;

-- Table: breeds
-- Stores detailed information about different cattle and buffalo breeds
CREATE TABLE IF NOT EXISTS breeds (
    breed_id INTEGER PRIMARY KEY AUTOINCREMENT,
    breed_name TEXT NOT NULL UNIQUE,
    breed_type TEXT NOT NULL CHECK (breed_type IN ('Cattle', 'Buffalo')),
    origin TEXT NOT NULL,
    description TEXT,
    milk_yield TEXT,
    fat_content TEXT,
    characteristics TEXT,
    purpose TEXT,
    weight_male TEXT,
    weight_female TEXT,
    special_features TEXT,
    conservation_status TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table: predictions
-- Stores prediction records from the model
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT NOT NULL,
    image_path TEXT,
    predicted_breed TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    top_predictions TEXT, -- JSON string containing top 3 predictions
    user_ip TEXT,
    user_agent TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (predicted_breed) REFERENCES breeds(breed_name)
);

-- Table: users
-- Stores user information (optional, for tracking purposes)
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_access DATETIME
);

-- Table: prediction_feedback
-- Stores user feedback on predictions
CREATE TABLE IF NOT EXISTS prediction_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    user_id INTEGER,
    is_correct BOOLEAN NOT NULL,
    correct_breed TEXT,
    comments TEXT,
    submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_breed ON predictions(predicted_breed);
CREATE INDEX IF NOT EXISTS idx_breeds_type ON breeds(breed_type);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Insert sample breed data (Indian cattle and buffalo breeds)
INSERT OR IGNORE INTO breeds (
    breed_name, breed_type, origin, description, milk_yield, fat_content, 
    characteristics, purpose, weight_male, weight_female, special_features, conservation_status
) VALUES 
('Gir', 'Cattle', 'Gujarat, India', 
 'A dairy cattle breed known for high milk production and distinctive curved horns. Gir cows are heat-tolerant and produce A2 milk.',
 '1,200 - 2,500 kg/lactation', '4.5 - 5.0%', 
 'Distinctive curved horns, loose skin, reddish brown to white color, medium to large size',
 'Dairy', '700 - 850 kg', '500 - 600 kg',
 'Heat tolerance, disease resistance, A2 beta-casein milk protein',
 'Endangered'),

('Sahiwal', 'Cattle', 'Punjab region (India/Pakistan)',
 'A heat-tolerant dairy breed with reddish brown to red coloration. Known for good milk production in tropical climates.',
 '1,400 - 2,800 kg/lactation', '4.5 - 5.0%',
 'Reddish brown to red color, loose skin, prominent hump, medium to large size',
 'Dairy', '650 - 800 kg', '450 - 550 kg',
 'Excellent heat tolerance, tick resistance',
 'Critical'),

('Ongole', 'Cattle', 'Andhra Pradesh, India',
 'A draught and dual-purpose breed known for bullock strength. White colored with distinctive hump.',
 '500 - 800 kg/lactation', '4.2 - 4.8%',
 'White color, massive hump, black pigmented skin, medium to large size',
 'Draught/Dual-purpose', '800 - 1,000 kg', '500 - 600 kg',
 'Excellent draught power, heat tolerance',
 'Critical'),

('Murrah', 'Buffalo', 'Haryana, India',
 'The most popular buffalo breed for milk production in India. Black colored with superior milk yield.',
 '1,500 - 2,500 kg/lactation', '7.0 - 8.0%',
 'Black color, small head, tightly curved horns, compact body',
 'Dairy', '650 - 800 kg', '500 - 650 kg',
 'Highest milk yield among buffalo breeds, good temperament',
 'Not endangered'),

('Jaffarabadi', 'Buffalo', 'Gujarat, India',
 'A large sized buffalo breed with black or spotted coloration. Good for both milk and draught purposes.',
 '1,200 - 2,000 kg/lactation', '6.5 - 7.5%',
 'Black with white spots or completely black, large size, massive build',
 'Dual-purpose', '750 - 900 kg', '600 - 750 kg',
 'Large size, good milk and meat production',
 'Endangered');

-- Create trigger to update the updated_at timestamp when a breed is modified
CREATE TRIGGER IF NOT EXISTS update_breed_timestamp
AFTER UPDATE ON breeds
FOR EACH ROW
BEGIN
    UPDATE breeds SET updated_at = CURRENT_TIMESTAMP WHERE breed_id = NEW.breed_id;
END;