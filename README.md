# 🐄 Cattle Breed Recognition System

**Image based breed recognition for cattle and buffaloes of India using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project implements an AI-powered system for recognizing Indian cattle and buffalo breeds from images using deep learning. The system uses transfer learning with ResNet-18 architecture to achieve high accuracy in breed classification.

### Key Features
- 🖼️ **Image Upload & Analysis**: Drag-and-drop interface for easy image upload
- 🤖 **Deep Learning Model**: ResNet-18 based transfer learning
- 🎯 **5 Breed Classification**: Ayrshire, Brown Swiss, Holstein Friesian, Jersey, Red Sindhi
- 💾 **Database Integration**: Store and retrieve prediction history
- 🌐 **RESTful API**: Complete backend API with multiple endpoints
- 📱 **Responsive UI**: Modern, mobile-friendly web interface
- 🚀 **Production Ready**: Docker support and deployment configuration

## 🏗️ Project Structure

```
cattle_ai_app/
├── backend/                # FastAPI backend application
│   ├── main.py            # Main API endpoints and routes
│   ├── server.py          # Server startup script
│   └── __init__.py
├── database/              # Database management
│   ├── db_manager.py      # SQLite database operations
│   └── __init__.py
├── models/                # ML model files
│   ├── model.py           # Model definition and inference
│   ├── model_training.py  # Training scripts
│   └── __init__.py
├── static/                # Static assets
│   ├── style.css          # Custom styles
│   └── uploads/           # Uploaded images
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── tests/                 # Test suite
│   └── test_api.py        # API tests
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── render.yaml           # Render deployment config
└── README.md             # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/abitha-balakrishnan/cattle_identification_project.git
cd cattle_ai_app
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Pre-trained Model
Place your trained model file (`cattle_model.pth`) in the project root directory, or train a new model using:

```bash
python models/model_training.py
```

## 🚀 Usage

### Starting the Server

#### Option 1: Using the server script
```bash
cd backend
python server.py --host 0.0.0.0 --port 8000
```

#### Option 2: Using uvicorn directly
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Application

Once the server is running, you can access:
- **Web Interface**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface home page |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Upload image and get breed prediction |
| `/breeds` | GET | Get list of all supported breeds |
| `/breeds/{breed_name}` | GET | Get detailed information about a specific breed |
| `/predictions` | GET | Get all predictions from database |
| `/predictions/{prediction_id}` | GET | Get specific prediction details |

### Example API Usage

#### Predict Breed from Image
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

#### Response Format
```json
{
  "image_url": "/static/uploads/your_image.jpg",
  "prediction": {
    "breed": "Holstein Friesian cattle",
    "confidence": 0.9234,
    "all_probabilities": {
      "Ayrshire cattle": 0.0123,
      "Brown Swiss cattle": 0.0234,
      "Holstein Friesian cattle": 0.9234,
      "Jersey cattle": 0.0345,
      "Red Sindhi cattle": 0.0064
    }
  },
  "timestamp": "2024-01-01T12:00:00",
  "id": "unique-prediction-id"
}
```

## 🎓 Model Information

### Architecture
- **Base Model**: ResNet-18 (Pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned on Indian cattle breed dataset
- **Input Size**: 256x256 pixels
- **Output Classes**: 5 cattle breeds

### Supported Breeds
1. **Ayrshire cattle** - Dairy breed from Scotland
2. **Brown Swiss cattle** - Ancient dairy breed from Switzerland
3. **Holstein Friesian cattle** - High-yield dairy breed from Netherlands
4. **Jersey cattle** - Small dairy breed from Jersey Island
5. **Red Sindhi cattle** - Indigenous dairy breed from Pakistan/India

### Performance Metrics
- **Accuracy**: ~85-90% (depends on dataset quality)
- **Inference Time**: < 1 second per image
- **Model Size**: ~45 MB

## 🗄️ Database

The application uses SQLite for storing predictions:

- **Database File**: `cattle_recognition.db` (auto-created)
- **Tables**: `predictions`
- **Stored Data**: Image URL, predicted breed, confidence score, timestamp

### View Stored Predictions
```bash
curl http://localhost:8000/predictions
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t cattle-breed-recognition .
```

### Run Docker Container
```bash
docker run -d -p 8000:8000 cattle-breed-recognition
```

### Deploy to Render
1. Push code to GitHub
2. Connect repository to Render
3. Use provided `render.yaml` configuration
4. Deploy automatically

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📊 Dataset Preparation

If you want to train the model yourself:

1. **Organize Images**: Create folder structure:
   ```
   dataset/
   ├── ayrshire_cattle/
   ├── brown_swiss_cattle/
   ├── holstein_friesian_cattle/
   ├── jersey_cattle/
   └── red_sindhi_cattle/
   ```

2. **Train Model**:
   ```bash
   python models/model_training.py --data_dir dataset --epochs 25
   ```

## 🔍 Troubleshooting

### Common Issues

**Issue**: Model not loading
```
Solution: Ensure cattle_model.pth is in the project root directory
```

**Issue**: Port already in use
```bash
Solution: Change port number or kill existing process
lsof -i :8000
kill -9 <PID>
```

**Issue**: Import errors
```bash
Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Database**: SQLite
- **ML Framework**: PyTorch, torchvision
- **Computer Vision**: OpenCV, Pillow
- **Deployment**: Docker, Render

## 📝 Development

### Adding New Breeds
1. Update `CLASS_NAMES` in `backend/main.py`
2. Add breed information in the `/breeds` endpoint
3. Retrain model with new dataset

### Customizing UI
- Edit `templates/index.html` for structure
- Modify `static/style.css` for styling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Indian Council of Agricultural Research (ICAR)
- National Bureau of Animal Genetic Resources (NBAGR)
- PyTorch and FastAPI communities

## 📧 Contact

For questions or support:
- **Email**: abithabalakrishnan2005@gmail.com
- **Project Link**: https://github.com/abitha-balakrishnan/cattle_identification_project

---

**Made with ❤️ for Indian Cattle & Buffalo Conservation**

*Supporting farmers and researchers with AI-powered breed identification*
