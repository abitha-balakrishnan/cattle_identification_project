#!/bin/bash

# Deployment Script for Indian Cattle & Buffalo Breed Recognition System

set -e  # Exit on any error

echo "🚀 Starting deployment for Cattle Breed Recognition System..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the project root."
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH. Please install Docker first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t cattle-breed-recognition .

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Run the container
echo "🏃 Running container on port 8000..."
docker run -d --name cattle-app -p 8000:8000 cattle-breed-recognition

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "✅ Container started successfully"
    echo "🌐 Application is running at http://localhost:8000"
    echo "🔧 Health check available at http://localhost:8000/health"
else
    echo "❌ Failed to start container"
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker logs -f cattle-app"
echo "   Stop container: docker stop cattle-app"
echo "   Remove container: docker rm cattle-app"
echo "   Remove image: docker rmi cattle-breed-recognition"
echo ""
echo "💡 To test the API:"
echo "   curl http://localhost:8000/health"
echo ""

# Wait a moment to let the server start
sleep 5

# Test the health endpoint
echo "🧪 Testing health endpoint..."
curl -s http://localhost:8000/health > /dev/null && echo "✅ Health check passed" || echo "⚠️ Health check failed - server may still be starting"

echo ""
echo "🚀 Deployment script completed!"