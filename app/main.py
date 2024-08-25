import os
import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the bloomz-1b1 model and tokenizer
model_name = "bigscience/bloomz-1b1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    do_sample: bool = True

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=request.max_length,
            do_sample=request.do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    # Calculate log probabilities
    log_probs = torch.nn.functional.log_softmax(output.scores[-1], dim=-1)
    generated_token_probs = torch.gather(log_probs, 1, output.sequences[:, -1].unsqueeze(-1))

    return {
        "generated_text": generated_text,
        "log_probabilities": generated_token_probs.cpu().numpy().tolist()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
