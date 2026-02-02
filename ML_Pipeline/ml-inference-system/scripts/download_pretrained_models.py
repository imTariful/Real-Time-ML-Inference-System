from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os

def export_model(model_id, output_path):
    print(f"Exporting {model_id} to {output_path}...")
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # V1: DistilBERT (faster)
    export_model("distilbert-base-uncased-finetuned-sst-2-english", "models/sentiment_v1")
    
    # V2: RoBERTa (more accurate, slightly slower - good for A/B test contrast)
    export_model("cardiffnlp/twitter-roberta-base-sentiment", "models/sentiment_v2")
