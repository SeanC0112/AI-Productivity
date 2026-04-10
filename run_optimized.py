#!/usr/bin/env python3
"""
Optimized Llama 3.2 inference with Metal acceleration
"""

import os
import torch
from pathlib import Path
from llama_models.llama3.generation import Llama3

# Distributed training setup for single machine
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")


def main():
    model_path = Path.home() / ".llama" / "checkpoints" / "Llama3.2-11B-Vision-Instruct"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print("📦 Loading Llama 3.2...\n")
    
    # Try different device options
    device = None
    
    # Try MPS first (may not work with all models)
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🖥️  Trying Metal Performance Shaders (MPS)...")
    except Exception as e:
        print(f"⚠️  MPS not available: {e}")
    
    # Fallback to CPU with optimizations
    if not device:
        device = "cpu"
        print("🖥️  Using CPU (optimized)")
    
    print()
    
    try:
        # Build generator
        generator = Llama3.build(
            ckpt_dir=str(model_path),
            max_seq_len=256,      # Reduced for faster inference on CPU
            max_batch_size=1,     # Single batch for CPU efficiency
            device=device,
            world_size=1,
        )
    except Exception as e:
        print(f"❌ Error loading with {device}: {e}")
        print("\n💡 Trying CPU as fallback...")
        try:
            device = "cpu"
            generator = Llama3.build(
                ckpt_dir=str(model_path),
                max_seq_len=256,
                max_batch_size=1,
                device=device,
                world_size=1,
            )
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return
    
    print("✅ Model loaded successfully!\n")
    print("="*60)
    print("Llama 3.2 Interactive Mode")
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
            batch = [prompt]
            for token_results in generator.completion(
                batch,
                temperature=0.7,
                top_p=0.9,
            ):
                result = token_results[0]
                if result.finished:
                    break
                print(result.text, end="", flush=True)
            
            print("\n" + "-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n(Interrupted)")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
