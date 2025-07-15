#!/usr/bin/env python3
"""
Setup script for RAG Engine development environment
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("✗ Python 3.9 or higher is required")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected")
    return True


def check_dependencies():
    """Check if required system dependencies are available"""
    dependencies = ["pip", "git"]
    
    for dep in dependencies:
        if shutil.which(dep) is None:
            print(f"✗ {dep} not found in PATH")
            return False
        print(f"✓ {dep} is available")
    
    return True


def create_virtual_environment():
    """Create a virtual environment"""
    if os.path.exists("venv"):
        print("✓ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")


def install_dependencies():
    """Install Python dependencies"""
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    commands = [
        (f"{pip_cmd} install --upgrade pip", "Upgrading pip"),
        (f"{pip_cmd} install -e .", "Installing rag-engine package"),
        (f"{pip_cmd} install -e .[dev]", "Installing development dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def setup_environment_file():
    """Setup .env file from template"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from template")
        print("  Please update .env file with your actual API keys")
        return True
    else:
        print("✗ .env.example template not found")
        return False


def create_data_directory():
    """Create data directory with .gitkeep"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    gitkeep = data_dir / ".gitkeep"
    gitkeep.touch()
    
    print("✓ Created data directory")
    return True


def check_chromadb_setup():
    """Check ChromaDB setup instructions"""
    print("\n" + "="*50)
    print("CHROMADB SETUP INSTRUCTIONS:")
    print("="*50)
    print("1. Install ChromaDB server (if not already installed):")
    print("   pip install chromadb")
    print("\n2. Start ChromaDB server:")
    print("   chroma run --host localhost --port 8000")
    print("\n3. Or use Docker:")
    print("   docker run -p 8000:8000 chromadb/chroma")
    print("\n4. Verify ChromaDB is running by visiting:")
    print("   http://localhost:8000/api/v1/heartbeat")
    print("="*50)


def main():
    """Main setup function"""
    print("RAG Engine Setup Script")
    print("=" * 30)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Setup steps
    setup_steps = [
        create_virtual_environment,
        install_dependencies,
        setup_environment_file,
        create_data_directory
    ]
    
    for step in setup_steps:
        if not step():
            print(f"\n✗ Setup failed during: {step.__name__}")
            sys.exit(1)
    
    # Final instructions
    print("\n" + "✓" * 30)
    print("Setup completed successfully!")
    print("✓" * 30)
    
    check_chromadb_setup()
    
    print("\nNext steps:")
    print("1. Update .env file with your OpenAI API key")
    print("2. Start ChromaDB server (see instructions above)")
    print("3. Run the RAG engine:")
    print("   python scripts/run_server.py")
    print("4. Test the API using requests.http file")


if __name__ == "__main__":
    main()