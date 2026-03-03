"""
Server Startup Script for Indian Cattle & Buffalo Breed Recognition System

This script provides a convenient way to start the FastAPI server with proper configuration.
"""

import uvicorn
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Start the cattle breed recognition API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Change to the backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    print(f"Starting Cattle Breed Recognition API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Auto-reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print(f"Backend Directory: {backend_dir.absolute()}")
    
    # Start the Uvicorn server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        timeout_keep_alive=300  # Increase timeout for ML model processing
    )

if __name__ == "__main__":
    main()