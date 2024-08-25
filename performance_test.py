import requests
import time
import random
import psutil
import torch
import asyncio
import aiohttp
from datasets import load_dataset

# Define the API URL (assuming the Docker container is running locally)
API_URL = "http://localhost:8000/generate"

async def fetch(session, prompt):
    async with session.post(API_URL, json={"prompt": prompt, "max_length": 100}) as response:
        return await response.json()

async def run_concurrent_requests(prompts, num_iterations=100, concurrency=10):
    times = []
    vram_usage = []
    cpu_usage = []

    async with aiohttp.ClientSession() as session:
        for i in range(num_iterations):
            prompt = prompts[i % len(prompts)]
            
            # Monitor CPU usage before API call
            cpu_before = psutil.cpu_percent(interval=None)
            
            # Measure VRAM usage before API call (GPU Memory usage)
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                vram_before = torch.cuda.memory_allocated()

            start_time = time.time()
            
            # Execute concurrent requests
            tasks = [fetch(session, prompt) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # Monitor CPU usage after API call
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Measure VRAM usage after API call (GPU Memory usage)
            if torch.cuda.is_available():
                vram_after = torch.cuda.memory_allocated()

            # Record the time taken for the requests
            inference_time = end_time - start_time
            times.append(inference_time / concurrency)  # Average time per request

            # Record VRAM and CPU usage
            if torch.cuda.is_available():
                vram_usage.append(vram_after - vram_before)
            cpu_usage.append((cpu_after + cpu_before) / 2)

            # Print details for each iteration
            print(f"Iteration {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Generated Texts: {[result.get('generated_text', '') for result in results]}")
            print(f"Inference Time (average per request): {inference_time / concurrency:.4f} seconds")
            if torch.cuda.is_available():
                print(f"VRAM Usage: {vram_usage[-1]/1024**2:.2f} MB")
            print(f"CPU Usage: {cpu_usage[-1]:.2f}%\n")

    # Calculate averages
    avg_time = sum(times) / len(times)
    max_time = max(times)
    avg_vram_usage = sum(vram_usage) / len(vram_usage) if vram_usage else None
    avg_cpu_usage = sum(cpu_usage) / len(cpu_usage)

    print(f"\nPerformance Test Results:")
    print(f"Average Time per Request: {avg_time:.4f} seconds")
    print(f"Maximum Time for a Single Request: {max_time:.4f} seconds")
    if avg_vram_usage is not None:
        print(f"Average VRAM Usage: {avg_vram_usage/1024**2:.2f} MB")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")

if __name__ == "__main__":
    # Load DailyDialog and Persona-Chat datasets
    dialog_dataset = load_dataset("daily_dialog", split="test")
    persona_dataset = load_dataset("persona_chat", split="validation")

    # Prepare a list of prompts from the datasets
    prompts = []

    # Add prompts from DailyDialog
    for sample in dialog_dataset:
        for dialog in sample["dialog"][:2]:  # Take the first two utterances as prompts
            prompts.append(dialog)

    # Add prompts from Persona-Chat
    for sample in persona_dataset:
        history = sample["history"]
        if len(history) > 1:
            prompt = history[-2]  # Use the second last turn as the prompt
            prompts.append(prompt)

    # Shuffle the prompts
    random.shuffle(prompts)

    # Run performance test with 100 API calls using concurrency
    asyncio.run(run_concurrent_requests(prompts, num_iterations=100, concurrency=10))
