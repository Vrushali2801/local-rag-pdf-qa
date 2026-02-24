"""
Quick test to verify Ollama connectivity and model response.
Run this to check if Ollama is working properly.
"""
import requests
import json
from config import OLLAMA_HOST, DEFAULT_MODEL, LLM_TIMEOUT

def test_ollama_connection():
    """Test basic Ollama API connectivity."""
    print("Testing Ollama connection...")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/version", timeout=5)
        response.raise_for_status()
        print(f"✅ Ollama is running: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {str(e)}")
        return False

def test_model_generation():
    """Test if the model can generate a simple response."""
    print(f"\nTesting {DEFAULT_MODEL} model generation...")
    
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": "Say hello!",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Model response: {result.get('response', '')[:100]}")
        return True
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {LLM_TIMEOUT} seconds")
        print("   Try increasing LLM_TIMEOUT in config.py")
        return False
    except Exception as e:
        print(f"❌ Model generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("OLLAMA CONNECTIVITY TEST")
    print("="*60)
    
    if test_ollama_connection():
        test_model_generation()
    
    print("\n" + "="*60)
    print("CONFIGURATION:")
    print(f"  Host: {OLLAMA_HOST}")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Timeout: {LLM_TIMEOUT}s")
    print("="*60)
