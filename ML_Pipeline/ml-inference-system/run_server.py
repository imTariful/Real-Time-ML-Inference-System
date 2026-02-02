#!/usr/bin/env python3
"""
Wrapper script to run the ML Inference System properly
This avoids path conflicts with other app.py files
"""
import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to the path at the beginning
sys.path.insert(0, script_dir)

if __name__ == "__main__":
    try:
        import uvicorn
        
        # Run the application
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8001,
            reload=False,
            log_level="info",
            workers=1
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
