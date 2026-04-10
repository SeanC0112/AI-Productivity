#!/usr/bin/env python3
"""
Simple script to query the Llama 3.2-11B-Vision-Instruct model
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
    # Path to the model checkpoint
    model_path = Path.home() / ".llama" / "checkpoints" / "Llama3.2-11B-Vision-Instruct"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please ensure the model is downloaded first.")
        return
    
    print(f"Loading model from {model_path}...")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device")
    elif torch.xpu.is_available():
        device = "xpu"
        print("Using XPU device")
    else:
        device = "cpu"
        print("Using CPU device (this will be slow)")
    
    # Initialize the generator
    generator = Llama3.build(
        ckpt_dir=str(model_path),
        max_seq_len=512,
        max_batch_size=4,
        device=device,
        world_size=1,
    )
    
    print("\n" + "="*60)
    print("Llama 3.2 Ready! Enter your prompt (type 'exit' to quit)")
    print("="*60 + "\n")
    
    # Interactive loop
    while True:
        prompt = input("\n🤖 Your prompt: ").strip()
        
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not prompt:
            print("Please enter a prompt.")
            continue
        
        print("\n✨ Response:")
        print("-" * 40)
        
        try:
            # Generate response
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
            print("\n" + "-" * 40)
        except Exception as e:
            print(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
