#!/usr/bin/env python3
"""
Setup Ollama for Llama 3.2 inference
Run this script to get started with Ollama
"""

import subprocess
import sys
import time
from pathlib import Path


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_ollama():
    """Instructions for installing Ollama"""
    print("📦 Ollama is not installed.")
    print("\n📥 Install Ollama from: https://ollama.ai")
    print("   Or run: brew install ollama")
    print("\n⏸️  After installing, come back and run this script again.")
    sys.exit(1)


def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def start_ollama_server():
    """Start the Ollama server"""
    print("🚀 Starting Ollama server...")
    try:
        # Try to start Ollama in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Check if it started
        if check_ollama_server():
            print("✅ Ollama server is running!")
            return True
        else:
            print("⚠️  Server didn't start. Try running manually:")
            print("   ollama serve")
            return False
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return False


def pull_model():
    """Pull the Llama 3.2 model"""
    print("\n📥 Downloading Llama 3.2 model (this may take a while)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", "llama3.2"],
            capture_output=False,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False


def main():
    print("="*60)
    print("Ollama Setup for Llama 3.2")
    print("="*60 + "\n")
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        install_ollama()
    
    print("✅ Ollama is installed!")
    
    # Check if server is running
    if not check_ollama_server():
        if not start_ollama_server():
            print("❌ Could not start Ollama server")
            sys.exit(1)
    
    # Pull model
    if not pull_model():
        print("❌ Failed to pull model")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✨ Setup complete!")
    print("="*60)
    print("\nYou can now use Ollama with:")
    print("   python run_with_ollama.py      (Simple chat)")
    print("   ollama run llama3.2             (CLI)")
    print("   python test_ollama.py           (Quick test)")
    print()


if __name__ == "__main__":
    main()
