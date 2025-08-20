import sys
import os
sys.path.append('src')

import openAIHandler
import openai
from dotenv import load_dotenv
import json
import pandas as pd
# Wait a moment and check status
import time
# Load environment variables
load_dotenv('src/.env')
import os
from groq import Groq
import json
from anthropic import Anthropic


# Execute both JSONL files using batch API
input_files = [
    'data/final_test/final/gpt-4o-mini/batch_1_gpt4o_mini.jsonl'
]

def check_batch_status_groq(batch_id):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    batch_status = client.batches.retrieve(batch_id)
    print(batch_status)
    return batch_status

def check_batch_status_openai(batch_id):
    client = openai.OpenAI()
    batch_status = client.batches.retrieve(batch_id)
    print(batch_status)
    return batch_status

def submit_claude_batch_requests(input_files):
    # Initialize client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for j, file_path in enumerate(input_files, 1):
        

        # Read and convert first 5 requests from JSONL file
        requests = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):

                data = json.loads(line.strip())
                
                request = {
                    "custom_id": data["custom_id"],
                    "params": {
                        "model": data["body"]["model"],
                        "max_tokens": 1000,

                        "messages": [
                            msg for msg in data["body"]["messages"] 
                            if msg["role"] != "system"
                        ],
                        "system": next(
                            (msg["content"] for msg in data["body"]["messages"] 
                                if msg["role"] == "system"), None
                        )
                    }
                }
                requests.append(request)

        print(f"Prepared {len(requests)} requests for batch processing")

        # Create the batch
        message_batch = client.messages.batches.create(requests=requests)
        print(f"Batch created: {message_batch.id}")
        print(f"Status: {message_batch.processing_status}")
    

def submit_groq_batch_requests(input_files):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    for i, file_path in enumerate(input_files, 1):
        print(f"\n📤 Submitting batch part {i}...")
        request_batch = client.files.create(file=open(file_path, "rb"), purpose="batch")
        print(request_batch)
        file_input_id = request_batch.id
        print(file_input_id)
        batch_job = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=file_input_id
        )
        print(batch_job)


def submit_openai_batch_requests(input_files):
    batch_jobs = []
    for i, input_file in enumerate(input_files, 1):
        print(f"\n📤 Submitting batch part {i}...")
        batch_job = openAIHandler.get_batch_job(input_file)
        batch_jobs.append(batch_job)
        
        print(f"✅ Batch part {i} submitted successfully!")
        print(f"  Batch ID: {batch_job.id}")
        print(f"  Status: {batch_job.status}")
        print(f"  Input file ID: {batch_job.input_file_id}")
        print(f"  Completion window: {batch_job.completion_window}")



    # Check the status of both batches
    client = openai.OpenAI()
    print(f"\n📊 Checking batch statuses...")

    for i, batch_job in enumerate(batch_jobs, 1):
        batch_status = client.batches.retrieve(batch_job.id)
        print(f"\nBatch part {i} (ID: {batch_job.id}):")
        print(f"  Status: {batch_status.status}")
        
        if batch_status.status == 'completed':
            print(f"  ✅ Batch part {i} completed! Output file ID: {batch_status.output_file_id}")
        elif batch_status.status == 'failed':
            print(f"  ❌ Batch part {i} failed!")
            if hasattr(batch_status, 'error'):
                print(f"    Error: {batch_status.error}")
        else:
            print(f"  ⏳ Batch part {i} still processing... Status: {batch_status.status}")

    print(f"\n📋 Summary:")
    print(f"  Total batches submitted: {len(batch_jobs)}")
    print(f"  Batch IDs: {[job.id for job in batch_jobs]}")
    print(f"  Use check_batch_status.py to download results when completed") 


if __name__ == "__main__":
    print("hi")
    input_files = [
       'data/extraction_results/claude_judge_input.jsonl'
    ]

    #batch_id_deepseek = 'batch_01k1kjx2kkf2a83md8tsfb4a4k'
    #batch_id_llama = 'batch_01k1y3bgwvfr4tsdynnb7mev2c'
    
    #submit_groq_batch_requests(input_files)
    #submit_openai_batch_requests(input_files)

    #openai_batch_id = 'batch_6891d17bf5f48190ae607a8762d15a3a'
    #llama_batch_id_1 = 'batch_01k1y8mwymebg81p0kwgxcrvp3'
    #llama_batch_id_2 = 'batch_01k1y6bwagewpbkzc2m6zah2w5'
    # deepseek_batch_id = 'batch_01k223gkrafe1tfaz9233g2vpe'

    submit_claude_batch_requests(input_files)

    #print(check_batch_status_groq(deepseek_batch_id))
    #print(check_batch_status_openai(openai_batch_id))