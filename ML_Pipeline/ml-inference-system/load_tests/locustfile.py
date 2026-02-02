from locust import HttpUser, task, between, constant
import random
import json

CORPUS = [
    "I love this product",
    "This is terrible",
    "Great service",
    "Worst experience ever",
    "Highly recommended",
    "Do not buy this",
]

class SentimentUser(HttpUser):
    wait_time = constant(0.1)  # Low wait time for high throughput

    @task(3)
    def predict_v1(self):
        text = random.choice(CORPUS)
        headers = {"x-token": "secret-token", "Content-Type": "application/json"}
        # Random ID
        req_id = str(random.randint(1, 1000000))
        payload = {
            "id": req_id,
            "texts": [text],
            "model_version": "v1"
        }
        self.client.post("/api/v1/predict", json=payload, headers=headers, name="/predict/v1")

    @task(1)
    def predict_v2(self):
        text = random.choice(CORPUS)
        headers = {"x-token": "secret-token", "Content-Type": "application/json"}
        req_id = str(random.randint(1, 1000000))
        payload = {
            "id": req_id,
            "texts": [text],
            "model_version": "v2"
        }
        self.client.post("/api/v1/predict", json=payload, headers=headers, name="/predict/v2")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
