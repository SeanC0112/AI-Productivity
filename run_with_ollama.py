#!/usr/bin/env python3
"""
Interactive chat with Llama 3.2 via Ollama
"""

import requests
import sys
from typing import Optional


def chat_with_ollama(prompt: str, model: str = "llama3.2") -> Optional[str]:
    """
    Send a prompt to Ollama and get a response
    
    Args:
        prompt: User's input
        model: Model name (default: llama3.2)
    
    Returns:
        Model's response, or None if error
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,  # 5 minute timeout for long responses
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server")
        print("   Make sure Ollama is running:")
        print("   1. Open a new terminal")
        print("   2. Run: ollama serve")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def check_ollama_running() -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def main():
    print("="*60)
    print("Llama 3.2 Chat with Ollama")
    print("="*60)
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("❌ Ollama server is not running!")
        print("\nTo start it, open a new terminal and run:")
        print("   ollama serve")
        print("\nThen come back here and try again.")
        sys.exit(1)
    
    print("\n✅ Connected to Ollama!\n")
    print("Type 'exit' to quit, 'clear' to clear history\n")
    print("-" * 60 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("🤖 You: ").strip()
            
            if user_input.lower() == "exit":
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "clear":
                conversation_history = []
                print("Conversation cleared.\n")
                continue
            
            if not user_input:
                continue
            
            # Build context from conversation history
            context = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in conversation_history
            ])
            
            if context:
                full_prompt = f"{context}\n\nUser: {user_input}\nAssistant:"
            else:
                full_prompt = user_input
            
            print("\n✨ Response:")
            response = chat_with_ollama(full_prompt)
            
            if response:
                print(response)
                
                # Store in history
                conversation_history.append({
                    "user": user_input,
                    "assistant": response.strip()
                })
                
                # Keep last 5 exchanges for context
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-5:]
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
