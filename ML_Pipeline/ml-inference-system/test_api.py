#!/usr/bin/env python3
"""Test the API endpoints"""
import sys
import requests
import json

BASE_URL = "http://127.0.0.1:8001"

print("=" * 60)
print("ML INFERENCE SYSTEM - API TESTS")
print("=" * 60)

# Test 1: Health endpoint
print("\n[TEST 1] Health Endpoint")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Prediction endpoint (should fail without auth token)
print("\n[TEST 2] Prediction Endpoint (No Auth)")
try:
    payload = {
        "id": "test-1",
        "texts": ["I love this product"],
        "model_version": "v1"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=payload,
        timeout=5
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Prediction endpoint (with auth token)
print("\n[TEST 3] Prediction Endpoint (With Auth)")
try:
    payload = {
        "id": "test-2",
        "texts": ["I love this product"],
        "model_version": "v1"
    }
    headers = {
        "x-token": "secret-token",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=payload,
        headers=headers,
        timeout=10
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Metrics endpoint
print("\n[TEST 4] Metrics Endpoint")
try:
    response = requests.get(f"{BASE_URL}/metrics", timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response Length: {len(response.text)} characters")
    print(f"First 300 chars:\n{response.text[:300]}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("TESTS COMPLETED")
print("=" * 60)
