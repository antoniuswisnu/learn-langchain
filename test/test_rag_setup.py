#!/usr/bin/env python3
"""
Test script to verify RAG agent setup
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_api_key():
    """Test if API key is set"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable is not set!")
        print("Please set it by running:")
        print('export GOOGLE_API_KEY="your-api-key-here"  # Unix/Linux/macOS')
        print('set GOOGLE_API_KEY=your-api-key-here       # Windows CMD')
        print('$env:GOOGLE_API_KEY="your-api-key-here"    # Windows PowerShell')
        print("\nOr add it to a .env file:")
        print('GOOGLE_API_KEY=your-api-key-here')
        return False
    else:
        print("✅ GOOGLE_API_KEY is set")
        return True

def test_imports():
    """Test if all required packages can be imported"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ langchain-google-genai imported successfully")
    except ImportError as e:
        print(f"❌ langchain-google-genai import failed: {e}")
        return False
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ langchain-huggingface imported successfully")
    except ImportError as e:
        print(f"❌ langchain-huggingface import failed: {e}")
        print("Install with: pip install langchain-huggingface")
        return False
    
    try:
        from langgraph.prebuilt import create_react_agent
        print("✅ langgraph imported successfully")
    except ImportError as e:
        print(f"❌ langgraph import failed: {e}")
        return False
    
    return True

def test_model():
    """Test if the model can be initialized"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ Google Generative AI model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Agent Setup...")
    print("=" * 50)
    
    imports_ok = test_imports()
    if not imports_ok:
        print("❌ Import tests failed. Install missing packages.")
        exit(1)
    
    api_key_ok = test_api_key()
    if not api_key_ok:
        print("❌ Please set up your GOOGLE_API_KEY first.")
        exit(1)
    
    model_ok = test_model()
    if model_ok:
        print("\n🎉 All tests passed! Your RAG agent should work now.")
        print("Run: python rag_agent.py")
    else:
        print("❌ Model test failed. Check your API key and internet connection.")