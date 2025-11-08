import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
# We are now importing BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional, Any

# --- Configuration ---
# This model is powerful and will load correctly on your 48GB L40S
#Qwen/Qwen3-32B
MODEL_ID = "meta-llama/Meta-Llama-3.1-70B"
DEVICE = "cuda"
PORT = 11434  # Use the same port as Ollama

print(f"Loading model {MODEL_ID} onto device {DEVICE}...")

# --- Load Model & Tokenizer with 4-bit BitsAndBytes ---

# Configure standard 4-bit quantization (This is the stable one)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,  # <-- Use the BitsAndBytes config
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded successfully. Starting server at http://localhost:{PORT}")

# --- API Setup ---
app = FastAPI()

# This class mimics the JSON payload that your OllamaAPI sends
class OllamaPayload(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

# This class mimics the JSON response your OllamaAPI expects
class OllamaResponse(BaseModel):
    response: str


@app.post("/api/generate", response_model=OllamaResponse)
async def generate_text(payload: OllamaPayload):
    """
    Handles the /api/generate request just like Ollama.
    """
    prompt = payload.prompt
    temperature = payload.options.get("temperature", 0.7)
    max_tokens = payload.options.get("num_predict", 3072)

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate text
    do_sample = True if temperature > 0.01 else False
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01), # temperature must be > 0
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode *only* the new tokens, skipping the prompt
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()

    # Return in the exact format the client expects
    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

    #echo 'export HF_HOME="/scratch1/huggingface"' >> ~/.bashrc