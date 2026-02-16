#!/usr/bin/env python3
"""
Test script to verify Mistral AI integration
Run this to make sure everything is working!
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("🧪 MISTRAL AI SETUP TEST")
print("=" * 60)

# Test 1: Check environment variables
print("\n✓ Test 1: Environment Variables")
print("-" * 60)

hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
model = os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

if hf_token:
    print(f"✅ HUGGINGFACE_API_TOKEN: {hf_token[:10]}...{hf_token[-5:]}")
else:
    print("❌ HUGGINGFACE_API_TOKEN not found in .env file!")
    print("   Get your FREE token from: https://huggingface.co/settings/tokens")
    exit(1)

print(f"✅ MISTRAL_MODEL: {model}")

# Test 2: Import Mistral Client
print("\n✓ Test 2: Import Mistral Client")
print("-" * 60)

try:
    from mistral_client import MistralClientWrapper
    print("✅ Successfully imported MistralClientWrapper")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print("   Make sure mistral_client.py is in the same directory")
    exit(1)

# Test 3: Initialize Client
print("\n✓ Test 3: Initialize Client")
print("-" * 60)

try:
    client = MistralClientWrapper(api_token=hf_token)
    print("✅ Successfully initialized Mistral client")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    exit(1)

# Test 4: Test API Call
print("\n✓ Test 4: Test API Call")
print("-" * 60)
print("Sending test message to Mistral AI...")
print("(First request may take 20-30 seconds while model loads)")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Mistral AI!' in exactly 5 words."}
        ],
        temperature=0.45,
        max_tokens=50
    )
    
    ai_response = response.choices[0].message.content
    print(f"\n✅ Mistral AI Response:\n   {ai_response}")
    
    # Check if model is loading
    if "model is loading" in ai_response.lower():
        print("\n⏳ Model is loading. Wait 20-30 seconds and run this test again!")
    else:
        print("\n🎉 SUCCESS! Mistral AI is working perfectly!")
    
except Exception as e:
    print(f"\n❌ API call failed: {e}")
    print("\nCommon issues:")
    print("- Invalid token: Check your HUGGINGFACE_API_TOKEN in .env")
    print("- Model loading: Wait 20-30 seconds and try again")
    print("- Network issue: Check your internet connection")
    exit(1)

# Test 5: Check Response Format
print("\n✓ Test 5: Response Format Compatibility")
print("-" * 60)

try:
    assert hasattr(response, 'choices'), "Missing 'choices' attribute"
    assert len(response.choices) > 0, "Empty choices list"
    assert hasattr(response.choices[0], 'message'), "Missing 'message' attribute"
    assert hasattr(response.choices[0].message, 'content'), "Missing 'content' attribute"
    print("✅ Response format is OpenAI-compatible")
except AssertionError as e:
    print(f"❌ Response format error: {e}")
    exit(1)

# Summary
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\nYour Mistral AI integration is ready to use!")
print("\nNext steps:")
print("1. Update your Flask app to use mistral_client")
print("2. Replace OpenAI imports with MistralClientWrapper")
print("3. Test your chat endpoints")
print("\nSee MIGRATION_GUIDE.md for detailed instructions.")
print("=" * 60)
