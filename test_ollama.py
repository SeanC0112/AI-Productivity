#!/usr/bin/env python3
"""
Quick test of Ollama with Llama 3.2
"""

import requests
import sys


def main():
    print("🧪 Testing Ollama connection...\n")
    
    # Check if Ollama server is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama server not responding")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to Ollama:")
        print(f"   {e}")
        print("\nMake sure Ollama is running:")
        print("   1. Open a new terminal")
        print("   2. Run: ollama serve")
        sys.exit(1)
    
    print("✅ Connected to Ollama server\n")
    
    # Check available models
    try:
        models_response = requests.get("http://localhost:11434/api/tags")
        models = models_response.json().get("models", [])
        
        if models:
            print("📦 Available models:")
            for model in models:
                print(f"   - {model['name']}")
        else:
            print("⚠️  No models found")
            print("   Download with: ollama pull llama3.2")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking models: {e}")
        sys.exit(1)
    
    print()
    
    # Test with a simple prompt
    test_prompt = "What is 2+2? Answer in one sentence."
    print(f"📝 Test prompt: {test_prompt}\n")
    print("💬 Response (streaming):")
    print("-" * 40)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": test_prompt,
                "stream": True,  # Use streaming to see progress
            },
            timeout=300,
            stream=True,
        )
        
        if response.status_code == 200:
            # Stream the response
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    print(data.get("response", ""), end="", flush=True)
            
            print("\n" + "-" * 40)
            print("\n✨ Ollama is working! Run:")
            print("   python run_with_ollama.py")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except KeyboardInterrupt:
        print("\n\n(Interrupted by user)")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
