#!/usr/bin/env python3
"""Simple test server runner"""
import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Python path: {sys.path[0]}")
    
    try:
        from app.main import app
        print("✓ FastAPI app imported successfully")
        
        import uvicorn
        print("✓ Starting uvicorn server on 127.0.0.1:8002...")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8002,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
