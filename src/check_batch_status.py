import openai
import os
from dotenv import load_dotenv
import json
import pandas as pd
from groq import Groq
from anthropic import Anthropic
# Load environment variables
load_dotenv('src/.env')
# Batch IDs from the new submission
batch_ids = [
    "batch_68873f67a2208190b04b71c6c51aac17"
    #"batch_688003ae01c48190a475499c12ef96bc",
    #"batch_688003c50a4c81908cfff66679e8caec"
]
import json

def combine_jsonl_files(input_files, output_file):
    """
    Combine multiple JSONL files into a single JSONL file.

    Args:
        input_files (list of str): List of input JSONL file paths.
        output_file (str): Output JSONL file path.
    """
    combined_records = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        combined_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON in {file_path}: {e}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in combined_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Combined {len(combined_records)} records into '{output_file}'.")

def analyze_paragraph_success_rate(results):
    """
    Analyze how many para_ids got at least 1 valid extraction.
    
    Args:
        results (list): List of parsed batch results
        
    Returns:
        dict: Analysis results
    """
    # Get unique para_ids from results
    unique_para_ids = set()
    para_ids_with_valid_phrases = set()
    
    for result in results:
        para_id = result.get('para_id', '')
        if para_id:
            unique_para_ids.add(para_id)
            
            # Check if this para_id has any valid phrases
            extracted_phrases = result.get('extracted_phrases', [])
            if extracted_phrases:
                para_ids_with_valid_phrases.add(para_id)
    
    total_para_ids = len(unique_para_ids)
    para_ids_with_valid = len(para_ids_with_valid_phrases)
    success_rate = (para_ids_with_valid / total_para_ids * 100) if total_para_ids > 0 else 0
    
    return {
        'total_para_ids': total_para_ids,
        'para_ids_with_valid_phrases': para_ids_with_valid,
        'para_ids_without_valid_phrases': total_para_ids - para_ids_with_valid,
        'success_rate': success_rate,
        'unique_para_ids': list(unique_para_ids),
        'para_ids_with_valid': list(para_ids_with_valid_phrases)
    }

def check_batch_status_groq(batch_id,output_path):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    batch_status = client.batches.retrieve(batch_id)
    print(batch_status)
    if batch_status.status == 'completed':
        result = client.files.content(batch_status.output_file_id)
        with open(output_path, 'wb') as f:
            f.write(result.read())
        print(f"📥 Results downloaded to: {output_path}")

        
    return batch_status

def check_batch_status_claude(batch_id,output_path):
    # Set up client
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    batch_status = client.messages.batches.retrieve(
        batch_id
    )
    all_results = []
    print(f"Status: {batch_status.processing_status=}")

    if batch_status.processing_status == 'ended':
        print("Now download it fron the claude website")
        
def check_batch_status_openai(batch_id,output_path):
# Set up client
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    all_results=[]
    batch_status = client.batches.retrieve(batch_id)
    print(f"Status: {batch_status.status}")

    if batch_status.status == 'completed':
        print(f"✅ Batch part {batch_id} completed! Output file ID: {batch_status.output_file_id}")
        
        # Download the results
        result = client.files.content(batch_status.output_file_id)
        
        with open(output_path, 'wb') as f:
            f.write(result.read())
        
        print(f"📥 Results downloaded to: {output_path}")
            
        # Parse and display the results
        print(f"📋 Parsing results for part {batch_id}...")
        part_results = []
        with open(output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Extract the response
                try:
                    response_content = data['response']['body']['choices'][0]['message']['content']
                    # Parse the JSON response
                    extracted_data = json.loads(response_content)
                    part_results.append(extracted_data)
                except Exception as e:
                    print(f"  Error parsing response: {e}")
                    continue

        print(f"✅ Parsed {len(part_results)} results for part {batch_id}")
        all_results.extend(part_results)
        
        # Show sample results
        if part_results:
            print(f"📄 Sample result from part {batch_id}:")
            sample_result = part_results[0]
            print(f"  para_id: {sample_result.get('para_id', 'MISSING')}")
            print(f"  section_id: {sample_result.get('section_id', 'MISSING')}")
            print(f"  extracted_phrases: {len(sample_result.get('extracted_phrases', []))} phrases")

    elif batch_status.status == 'failed':
        print(f"❌ Batch part {batch_id} failed!")
        if hasattr(batch_status, 'error'):
            print(f"Error: {batch_status.error}")

    else:
        print(f"⏳ Batch part {batch_id} still processing... Status: {batch_status.status}")
        print("Check again in a few minutes.")

    # If we have results from both batches, analyze them together
    if all_results:
        print(f"\n{'='*50}")
        print(f"📊 COMBINED ANALYSIS")
        print(f"{'='*50}")
        
        # Analyze paragraph success rate for all results
        print("📊 Analyzing paragraph success rate...")
        analysis = analyze_paragraph_success_rate(all_results)
        
        print(f"\n📈 Combined Paragraph Success Analysis:")
        print(f"  Total unique para_ids: {analysis['total_para_ids']}")
        print(f"  Para_ids with at least 1 valid phrase: {analysis['para_ids_with_valid_phrases']}")
        print(f"  Para_ids with no valid phrases: {analysis['para_ids_without_valid_phrases']}")
        print(f"  Success rate: {analysis['success_rate']:.1f}%")
        
        if analysis['para_ids_without_valid_phrases'] > 0:
            print(f"\n❌ Para_ids with no valid phrases:")
            for para_id in analysis['unique_para_ids']:
                if para_id not in analysis['para_ids_with_valid']:
                    print(f"    - {para_id}")

        # Show sample results from combined data
        print(f"\n📄 Sample results from combined data:")
        for i, result in enumerate(all_results[:3], 1):
            print(f"\n  Result {i}:")
            print(f"    para_id: {result.get('para_id', 'MISSING')}")
            print(f"    section_id: {result.get('section_id', 'MISSING')}")
            print(f"    extracted_phrases: {len(result.get('extracted_phrases', []))} phrases")
            
            # Show first phrase if available
            if result.get('extracted_phrases'):
                first_phrase = result['extracted_phrases'][0]
                print(f"    First phrase:")
                print(f"      case_law_excerpt: {first_phrase.get('case_law_excerpt', 'MISSING')[:100]}...")
                print(f"      legislation_excerpt: {first_phrase.get('legislation_excerpt', 'MISSING')[:100]}...")
                print(f"      confidence: {first_phrase.get('confidence', 'MISSING')}")

    print(f"\n📋 Summary:")
    print(f"  Total batches checked: {len(batch_ids)}")
    print(f"  Batch IDs: {batch_ids}")
    print(f"  Total results processed: {len(all_results)}") 

if __name__ == "__main__":
    print("hi")
    openai_batch_id = 'batch_689a1565d1bc8190ad81cdf5ef80b8f9'
    #llama_batch_id = 'batch_01k2csn1szfj3bd7eg5t6d8ms2'
    #llama_batch_id_2 = 'batch_01k1yc9c74f84bpzxasabszqk6'
    # deepseek_batch_id = 'batch_01k2afnh32f2p9dxttwwfwrj8h'

    output_path = 'data/final_test/final/reexperiment/fewhot/11August/redo-gpt-4o-mini_output-03.jsonl'
    #check_batch_status_groq(llama_batch_id,output_path)
    #check_batch_status_openai(openai_batch_id, output_path)

    # msg_batch_id = 'msgbatch_01UyoSrX5mts2vZojuu3FdGC'
    # output_path_claude = 'data/final_test/final/reexperiment/fewhot/output_batches/sample_claude_output.jsonl'
    # check_batch_status_claude(msg_batch_id, output_path_claude)

    files_to_combine = [
       'data/final_test/final/reexperiment/fewhot/11August/redo-gpt-4o-mini_output-01.jsonl',
       'data/final_test/final/reexperiment/fewhot/11August/redo-gpt-4o-mini_output-02.jsonl',
       'data/final_test/final/reexperiment/fewhot/11August/redo-gpt-4o-mini_output-03.jsonl'
      ]
    output_path = 'data/final_test/final/reexperiment/fewhot/11August/redo-gpt-4o-mini_output-combined.jsonl'
    combine_jsonl_files(files_to_combine, output_path)
