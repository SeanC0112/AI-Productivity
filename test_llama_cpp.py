#!/usr/bin/env python3
"""
Quick test of llama.cpp setup
"""

from pathlib import Path
from llama_cpp import Llama


def main():
    model_dir = Path.home() / ".llama" / "gguf"
    model_file = model_dir / "Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf"
    
    if not model_file.exists():
        print(f"❌ Model not found. Run conversion first:")
        print("   python convert_to_gguf.py")
        return
    
    print("📦 Testing llama.cpp...")
    
    llm = Llama(
        model_path=str(model_file),
        n_ctx=512,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )
    
    print("✅ Model loaded!\n")
    
    # Quick test
    test_prompt = "What is 2+2?"
    print(f"📝 Test: {test_prompt}")
    print("\n💬 Response:")
    print("-" * 40)
    
    response = llm(
        test_prompt,
        max_tokens=150,
        temperature=0.7,
    )
    
    print(response["choices"][0]["text"])
    print("-" * 40)
    print("\n✨ llama.cpp is working! Run 'python run_llama_cpp.py' for interactive mode.")


if __name__ == "__main__":
    main()
