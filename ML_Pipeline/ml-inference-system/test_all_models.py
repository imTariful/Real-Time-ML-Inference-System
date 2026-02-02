#!/usr/bin/env python3
"""Test all 5 pretrained models"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"

test_sentences = [
    "I absolutely love this product!",
    "This is terrible and I hate it.",
    "It's okay, nothing special.",
    "Amazing work! Highly recommended!",
    "Worst experience ever."
]

print("=" * 80)
print("TESTING 5 PRETRAINED SENTIMENT MODELS")
print("=" * 80)

# Get available models
print("\n[STEP 1] Fetching available models...")
try:
    r = requests.get(f"{BASE_URL}/api/v1/models")
    if r.status_code == 200:
        models_data = r.json()
        print(f"✓ Found {models_data['total']} models")
        for model in models_data['models']:
            print(f"  - {model['version']:3} | {model['name']:30} | {model['model_id']}")
    else:
        print(f"✗ Error: {r.status_code}")
except Exception as e:
    print(f"✗ Connection error: {e}")
    exit(1)

# Test each model
models = ["v1", "v2", "v3", "v4", "v5"]

for model_version in models:
    print(f"\n{'='*80}")
    print(f"[TESTING] Model {model_version.upper()}")
    print(f"{'='*80}")
    
    try:
        payload = {
            "id": f"test-{model_version}",
            "texts": test_sentences[:2],  # Test first 2 sentences
            "model_version": model_version
        }
        
        headers = {"x-token": "secret-token"}
        
        print(f"Sending request...")
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=payload,
            headers=headers,
            timeout=30
        )
        duration = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            print(f"✓ Status: {r.status_code} (Response time: {duration:.2f}s)")
            print(f"✓ Model Version: {result['model_version']}")
            print(f"✓ Latency: {result['latency_ms']:.2f}ms")
            print(f"✓ Results:")
            for i, pred in enumerate(result['results']):
                print(f"   [{i+1}] '{pred['text']}'")
                print(f"       Label: {pred['class']} (confidence: {pred['confidence']:.2%})")
                if 'raw_label' in pred:
                    print(f"       Raw Label: {pred['raw_label']}")
        else:
            print(f"✗ Error: {r.status_code}")
            print(f"Response: {r.text}")
    except requests.exceptions.Timeout:
        print(f"✗ Timeout: Model took too long to load")
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Connection error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
