#!/usr/bin/env python3
"""
Quick test to verify the model loads and can generate text
"""

import os
import torch
from pathlib import Path
from llama_models.llama3.generation import Llama3

# Set up distributed training environment variables for single-machine inference
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")


def main():
    model_path = Path.home() / ".llama" / "checkpoints" / "Llama3.2-11B-Vision-Instruct"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print("📦 Loading model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
    print(f"🖥️  Using {device.upper()} device")
    
    # Build generator
    generator = Llama3.build(
        ckpt_dir=str(model_path),
        max_seq_len=256,
        max_batch_size=4,
        quantization_mode="8bit",
        device=device,
        world_size=1,
    )
    
    print("✅ Model loaded successfully!\n")
    
    # Test prompt
    test_prompt = "What is the capital of France?"
    print(f"📝 Test prompt: {test_prompt}")
    print("\n💬 Response:")
    print("-" * 40)
    
    batch = [test_prompt]
    for token_results in generator.completion(batch, temperature=0.7, top_p=0.9):
        result = token_results[0]
        if result.finished:
            break
        print(result.text, end="", flush=True)
    
    print("\n" + "-" * 40)
    print("\n✨ Model is working! Run 'python query_llama.py' for interactive mode.")


if __name__ == "__main__":
    main()
