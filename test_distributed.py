import requests
import time

MASTER_URL = "http://localhost:8000"

def test_distributed():
    print("Submitting Job...")
    payload = {
        "type": "action", 
        "prompt": "Valid actions:\n- discard 3 of Hearts\n- discard 4 of Spades\nPick one.",
        "options": {},
        "priority": "fast"
    }
    
    try:
        start = time.time()
        res = requests.post(f"{MASTER_URL}/api/generate", json=payload, timeout=10)
        print(f"Response: {res.json()}")
        print(f"Time: {time.time() - start:.2f}s")
        if "response" in res.json():
            print("SUCCESS: Distributed System works.")
        else:
            print("FAILURE: Invalid response.")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_distributed()
