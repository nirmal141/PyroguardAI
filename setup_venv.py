#!/usr/bin/env python3
"""
Automated setup script for PyroGuard AI virtual environment.
This script creates a virtual environment and installs all required dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Set up the PyroGuard AI environment."""
    print("🔥 PyroGuard AI - Automated Environment Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found!")
        print("Please run this script from the PyroGuard AI project directory.")
        sys.exit(1)
    
    # Create virtual environment
    if not run_command("python -m venv pyroguard_env", 
                      "Creating virtual environment"):
        sys.exit(1)
    
    # Determine activation script path based on OS
    if os.name == 'nt':  # Windows
        activate_script = "pyroguard_env\\Scripts\\activate"
        pip_command = "pyroguard_env\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "pyroguard_env/bin/activate"
        pip_command = "pyroguard_env/bin/pip"
    
    # Install requirements
    install_cmd = f"{pip_command} install -r requirements.txt"
    if not run_command(install_cmd, "Installing Python packages"):
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print(f"   pyroguard_env\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print(f"   source pyroguard_env/bin/activate")
    
    print("\n2. Run the demo:")
    print("   python demo.py")
    
    print("\n3. When done, deactivate with:")
    print("   deactivate")
    
    print("\n🔥 Ready to fight fires with AI!")

if __name__ == "__main__":
    main()
