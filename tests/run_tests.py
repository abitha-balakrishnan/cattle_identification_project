"""
Test Runner for Indian Cattle & Buffalo Breed Recognition System

This script runs all tests in the project and provides a summary of results.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import time
from datetime import datetime


def run_tests(test_directory=".", test_pattern="test_*.py", verbose=False):
    """
    Run tests using pytest.
    
    Args:
        test_directory: Directory containing test files
        test_pattern: Pattern to match test files
        verbose: Whether to run tests in verbose mode
    """
    cmd = ["pytest", test_directory]
    
    if verbose:
        cmd.append("-v")
    
    # Add more detailed output
    cmd.extend(["-v", "--tb=short", "--strict-markers"])
    
    print(f"Running tests in {test_directory} with pattern {test_pattern}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("TEST RUN COMPLETE")
        print("="*60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest using 'pip install pytest'")
        return False
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return False


def run_specific_test_suite(suite_name):
    """
    Run a specific test suite.
    
    Args:
        suite_name: Name of the test suite (api, database, model, all)
    """
    suites = {
        "api": "test_api.py",
        "database": "test_database.py", 
        "model": "test_model.py",
        "all": "*test*.py"
    }
    
    if suite_name not in suites:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {list(suites.keys())}")
        return False
    
    pattern = suites[suite_name]
    return run_tests(test_pattern=pattern, verbose=True)


def install_test_dependencies():
    """Install required test dependencies."""
    print("Installing test dependencies...")
    
    try:
        # Check if requirements file exists
        req_file = Path("requirements-test.txt")
        if req_file.exists():
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        else:
            # Install basic test dependencies
            cmd = [sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "requests", "httpx", "fastapi"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Test dependencies installed successfully!")
            return True
        else:
            print("Error installing dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        return False


def generate_test_report():
    """Generate a test report."""
    print("Generating test report...")
    
    # Run tests with coverage and junit xml output
    cmd = [
        "pytest", 
        ".", 
        "-v", 
        "--cov=.", 
        "--cov-report=html:coverage_report", 
        "--cov-report=term-missing",
        "--junitxml=test-results.xml",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\nTest report generated!")
        print("- Coverage report: coverage_report/index.html")
        print("- JUnit XML: test-results.xml")
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Runner for Cattle Breed Recognition System")
    parser.add_argument("suite", nargs="?", default="all", 
                       help="Test suite to run (api, database, model, all)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Run tests in verbose mode")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install test dependencies")
    parser.add_argument("--report", action="store_true", 
                       help="Generate test report with coverage")
    parser.add_argument("--pattern", type=str, default="test_*.py",
                       help="Pattern for test files")
    
    args = parser.parse_args()
    
    # Change to tests directory
    tests_dir = Path(__file__).parent
    os.chdir(tests_dir)
    
    print(f"Cattle Breed Recognition System - Test Runner")
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests directory: {tests_dir.absolute()}")
    print("-" * 60)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            print("Failed to install dependencies. Exiting.")
            sys.exit(1)
    
    # Run tests
    success = True
    if args.report:
        success = generate_test_report()
    elif args.suite != "all":
        success = run_specific_test_suite(args.suite)
    else:
        success = run_tests(test_pattern=args.pattern, verbose=args.verbose)
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()