#!/usr/bin/env python3
"""Minimal test to identify the issue"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print("Step 1: Importing FastAPI...")
from fastapi import FastAPI
print("✓ FastAPI imported")

print("Step 2: Creating app...")
app = FastAPI()
print("✓ App created")

print("Step 3: Adding routes...")
@app.get("/health")
async def health():
    return {"status": "ok"}
print("✓ Route added")

print("Step 4: Starting server...")
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
