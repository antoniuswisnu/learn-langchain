#!/usr/bin/env python3
"""
Test script to check if the setup is working
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def test_api_key():
    """Test if API key is set"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable is not set!")
        print("Please set it by running:")
        print('export GOOGLE_API_KEY="your-api-key-here"')
        print("Or on Windows:")
        print('set GOOGLE_API_KEY=your-api-key-here')
        return False
    else:
        print("✅ GOOGLE_API_KEY is set")
        return True

def test_model():
    """Test if the model can be initialized"""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ Model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LangChain + Google Generative AI setup...")
    print("=" * 50)
    
    api_key_ok = test_api_key()
    if api_key_ok:
        model_ok = test_model()
        if model_ok:
            print("✅ All tests passed! You're ready to run the quickstart.")
        else:
            print("❌ Model test failed. Check your API key.")
    else:
        print("❌ Please set up your GOOGLE_API_KEY first.")