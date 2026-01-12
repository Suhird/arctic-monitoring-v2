"""
Test ML API Endpoints
"""
from fastapi.testclient import TestClient
from app.main import app
import os
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def test_classify_endpoint():
    print("\nTesting /api/v1/ml/classify...")
    
    # Create a dummy image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    response = client.post(
        "/api/v1/ml/classify",
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_prediction_endpoint():
    print("\nTesting /api/v1/predictions/7day?use_ml=true...")
    
    response = client.get(
        "/api/v1/predictions/7day",
        params={
            "min_lon": -160, "min_lat": 70, 
            "max_lon": -150, "max_lat": 75,
            "use_ml": "true"
        }
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Features: {len(data['features'])}")
        print(f"First feature props: {data['features'][0]['properties']}")
    else:
        print(response.text)
        
    assert response.status_code == 200
    assert len(response.json()['features']) == 7

def test_changes_endpoint():
    print("\nTesting /api/v1/ice/changes/detect...")
    
    response = client.get(
        "/api/v1/ice/changes/detect",
        params={
            "bbox": "-160,70,-150,75",
            "date1": "2023-01-01",
            "date2": "2023-01-08"
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    assert "change_detected_percent" in response.json()

if __name__ == "__main__":
    test_classify_endpoint()
    test_prediction_endpoint()
    test_changes_endpoint()
