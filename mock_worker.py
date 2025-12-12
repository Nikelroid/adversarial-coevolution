import requests
import time
import argparse
import random

def run_mock_worker(master_url, worker_type, worker_id):
    print(f"Mock Worker {worker_id} ({worker_type}) connecting to {master_url}")
    
    # Register
    requests.post(f"{master_url}/register_worker", json={
        "worker_id": worker_id, 
        "worker_type": worker_type,
        "model_id": "mock-model"
    })
    
    while True:
        try:
            # Poll
            response = requests.get(f"{master_url}/get_job", params={"worker_type": worker_type})
            if response.status_code == 200:
                job = response.json()
                if job:
                    print(f"[{worker_type}] Processing job {job['job_id']}")
                    prompt = job['payload']['prompt']
                    
                    # Generate Mock Response
                    if worker_type == "fast": # Action
                         # Just pick a random action from prompt parsing or dummy
                         # We'll return a valid-looking string
                         # Prompt contains "Valid actions:\n- discard 3 of Hearts\n..."
                         lines = prompt.split('\n')
                         options = [l.strip()[2:] for l in lines if l.strip().startswith("- ")]
                         if options:
                             res = random.choice(options)
                         else:
                             res = "discard 2 of Spades"
                    else: # Enhance
                        res = "You are an improved Gin Rummy agent. Focus on low deadwood."
                        
                    # Submit
                    requests.post(f"{master_url}/submit_result", json={
                        "job_id": job['job_id'],
                        "result": res,
                        "worker_id": worker_id,
                        "duration": 0.1
                    })
                else:
                    time.sleep(0.5)
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="fast")
    args = parser.parse_args()
    run_mock_worker("http://localhost:8000", args.type, f"mock_{args.type}")
