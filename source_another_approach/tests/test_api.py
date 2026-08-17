
import os, json
from fastapi.testclient import TestClient
from shop_ml.serve import app

def test_health():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert "status" in r.json()
