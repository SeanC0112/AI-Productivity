#!/usr/bin/env python3
"""
Run Llama 3.2 using llama.cpp (fast inference on CPU/Mac)
"""

from pathlib import Path
from llama_cpp import Llama


def main():
    model_dir = Path.home() / ".llama" / "gguf"
    model_file = model_dir / "Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf"
    
    if not model_file.exists():
        print(f"❌ Model not found at {model_file}")
        print("\nFirst run the conversion:")
        print("  python convert_to_gguf.py")
        return
    
    print("📦 Loading GGUF model with llama.cpp...")
    print(f"   Model: {model_file.name}")
    
    # Initialize llama.cpp
    # n_gpu_layers: on Mac with Metal, use -1 to offload all to GPU
    llm = Llama(
        model_path=str(model_file),
        n_ctx=2048,           # Context window
        n_threads=8,          # CPU threads (match your core count)
        n_gpu_layers=-1,      # Offload to Metal/GPU if available
        verbose=False,
    )
    
    print("✅ Model loaded successfully!\n")
    print("="*60)
    print("Llama 3.2 Interactive Mode (llama.cpp)")
    print("Type 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        prompt = input("🤖 Your prompt: ").strip()
        
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not prompt:
            print("Please enter a prompt.\n")
            continue
        
        print("\n✨ Response:")
        print("-" * 40)
        
        try:
            # Generate response
            response = llm(
                prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "\n\n"],
                echo=False,
            )
            
            print(response["choices"][0]["text"])
            print("-" * 40 + "\n")
            
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
