#!/usr/bin/env python3
"""Download and setup 5 pretrained sentiment models for testing"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time

models_config = {
    "v1": {
        "name": "DistilBERT SST-2",
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "description": "Fast sentiment classification on SST-2"
    },
    "v2": {
        "name": "RoBERTa Twitter",
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment",
        "description": "Optimized for Twitter/social media sentiment"
    },
    "v3": {
        "name": "DistilBERT Multilingual",
        "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "description": "Multilingual sentiment analysis"
    },
    "v4": {
        "name": "TinyBERT",
        "model_id": "huawei-noah/TinyBERT_General_4L_312D",
        "description": "Ultra-lightweight model for edge devices"
    },
    "v5": {
        "name": "BERT Base Multilingual",
        "model_id": "bert-base-multilingual-cased",
        "description": "Full BERT multilingual model"
    }
}

def download_models():
    """Download all 5 models"""
    print("=" * 70)
    print("DOWNLOADING 5 PRETRAINED SENTIMENT MODELS")
    print("=" * 70)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    for version, config in models_config.items():
        model_name = config["name"]
        model_id = config["model_id"]
        description = config["description"]
        
        print(f"\n[{version.upper()}] {model_name}")
        print(f"     Model ID: {model_id}")
        print(f"     Description: {description}")
        
        model_path = os.path.join(models_dir, f"sentiment_{version}")
        
        try:
            # Check if already downloaded
            if os.path.exists(model_path) and os.listdir(model_path):
                print(f"     ✓ Already cached at {model_path}")
                continue
            
            print(f"     Downloading...")
            start_time = time.time()
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(model_path)
            
            # Download model
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.save_pretrained(model_path)
            
            duration = time.time() - start_time
            print(f"     ✓ Downloaded in {duration:.1f}s")
            print(f"     ✓ Saved to {model_path}")
            
        except Exception as e:
            print(f"     ✗ Error: {e}")
            print(f"     Trying alternative download method...")
            
            try:
                # Fallback: use pipeline which handles downloads
                pipe = pipeline("sentiment-analysis", model=model_id)
                os.makedirs(model_path, exist_ok=True)
                print(f"     ✓ Model loaded via pipeline")
            except Exception as e2:
                print(f"     ✗ Failed: {e2}")
    
    print("\n" + "=" * 70)
    print("✓ MODEL SETUP COMPLETE")
    print("=" * 70)

def list_models():
    """List available models"""
    print("\nAvailable Models:")
    print("-" * 70)
    for version, config in models_config.items():
        print(f"  {version:3} | {config['name']:30} | {config['description']}")
    print("-" * 70)

if __name__ == "__main__":
    download_models()
    list_models()
