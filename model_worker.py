import torch
import time
import requests
import argparse
import sys
# We are now importing BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.config import get_config

CONFIG = get_config()
MODEL_MAP = CONFIG.get("models", {})
DIST_CONFIG = CONFIG.get("distributed", {}).get("worker", {})
DEFAULT_MODEL = DIST_CONFIG.get("default_model", "llama-3b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelWorker:
    def __init__(self, master_url, model_id, worker_type, worker_id):
        self.master_url = master_url
        self.model_id = model_id
        self.worker_type = worker_type # 'fast' or 'slow' (enhancer)
        self.worker_id = worker_id
        
        print(f"[Worker {worker_id}] Loading model {model_id} onto device {DEVICE}...")
        self._load_model()
        print(f"[Worker {worker_id}] Model loaded. Connecting to Master at {master_url}...")

    def _load_model(self):
        # Configure standard 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def generate(self, prompt, options=None):
        if options is None:
            options = {}
            
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("num_predict", 3072)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        do_sample = True if temperature > 0.01 else False
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response_text

    def register(self):
        try:
            payload = {
                "worker_id": self.worker_id,
                "worker_type": self.worker_type,
                "model_id": self.model_id
            }
            requests.post(f"{self.master_url}/register_worker", json=payload)
            print("[Registered]")
        except Exception as e:
            print(f"[Error Registering] {e}")

    def run(self):
        self.register()
        
        while True:
            try:
                # Poll for job
                response = requests.get(f"{self.master_url}/get_job", params={"worker_type": self.worker_type})
                
                if response.status_code == 200:
                    job = response.json()
                    if job:
                        job_id = job['job_id']
                        print(f"[Job {job_id}] Received job: {job['type']}")
                        
                        # Process
                        prompt = job['payload']['prompt']
                        options = job['payload'].get('options', {})
                        
                        start = time.time()
                        result_text = self.generate(prompt, options)
                        duration = time.time() - start
                        
                        # Submit result
                        result_payload = {
                            "job_id": job_id,
                            "result": result_text,
                            "worker_id": self.worker_id,
                            "duration": duration
                        }
                        requests.post(f"{self.master_url}/submit_result", json=result_payload)
                        print(f"[Job {job_id}] Completed in {duration:.2f}s")
                    else:
                        # Queue empty
                        time.sleep(1)
                else:
                    print(f"Master error: {response.text}")
                    time.sleep(5)
                    
            except requests.exceptions.ConnectionError:
                print("Connection lost. Retrying...")
                time.sleep(5)
            except Exception as e:
                print(f"Worker Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    master_default = CONFIG.get("distributed", {}).get("master", {}).get("url", "http://localhost:8000")
    
    parser.add_argument("--master", default=master_default)
    # Get model ID from map if possible, else use raw string
    default_model_id = MODEL_MAP.get(DEFAULT_MODEL, DEFAULT_MODEL)
    
    parser.add_argument("--model", default=default_model_id)
    parser.add_argument("--type", default="slow", choices=["fast", "slow"]) 
    parser.add_argument("--id", default="worker_1")
    args = parser.parse_args()
    
    worker = ModelWorker(args.master, args.model, args.type, args.id)
    worker.run()
