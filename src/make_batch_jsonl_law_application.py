import pandas as pd
import json
import os



import json
import os
import pandas as pd
from typing import List, Dict, Tuple
from math import ceil

def detect_provider_from_model(model_name: str) -> str:
    """
    Detect the provider based on model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Provider name ('openai' or 'groq')
    """
    groq_models = [
        'deepseek-r1-distill-llama-70b',
        'llama-3.3-70b-versatile',
        'mistral-saba-24b', 
        'llama-3.1-8b-instant',
        'meta-llama/llama-4-scout-17b-16e-instruct',
        'meta-llama/llama-4-maverick-17b-128e-instruct',
        'meta-llama/llama-guard-4-12b'
        'claude-sonnet-4-20250514'
    ]
    
    openai_models = [
        'gpt-4o-mini',
        'gpt-4o',
        'gpt-3.5-turbo',
        'gpt-4-turbo'
    ]
    
    # Check exact matches first
    if model_name in groq_models:
        return 'groq'
    elif model_name in openai_models:
        return 'openai'
    
    # Check partial matches for model families
    if any(groq_model in model_name for groq_model in ['llama', 'deepseek', 'mistral']):
        return 'groq'
    elif any(openai_model in model_name for openai_model in ['gpt', 'chatgpt']):
        return 'openai'
    
    # Default to OpenAI if uncertain
    print(f"Warning: Could not detect provider for model '{model_name}'. Defaulting to OpenAI limits.")
    return 'openai'

def get_batch_limits(provider: str) -> Dict:
    """
    Get batch API limits for the provider.
    
    Args:
        provider: 'openai' or 'groq'
        
    Returns:
        Dictionary with limits
    """
    limits = {
        'openai': {
            'max_file_size_mb': 100,
            'max_requests': 50000,
            'provider_name': 'OpenAI'
        },
        'groq': {
            'max_file_size_mb': 200,
            'max_requests': 50000,
            'provider_name': 'Groq'
        }
    }
    
    return limits.get(provider, limits['openai'])

def estimate_file_size_mb(jsonl_lines: List[Dict]) -> float:
    """
    Estimate the file size in MB for JSONL lines.
    
    Args:
        jsonl_lines: List of JSON objects
        
    Returns:
        Estimated file size in MB
    """
    # Calculate average line size by sampling
    sample_size = min(100, len(jsonl_lines))
    total_chars = 0
    
    for i in range(0, sample_size):
        line_json = json.dumps(jsonl_lines[i])
        total_chars += len(line_json) + 1  # +1 for newline
    
    avg_chars_per_line = total_chars / sample_size
    total_chars_estimate = avg_chars_per_line * len(jsonl_lines)
    
    # Convert to MB (assuming UTF-8 encoding, roughly 1 byte per character)
    size_mb = total_chars_estimate / (1024 * 1024)
    
    return size_mb

def calculate_splits_needed(total_requests: int, estimated_size_mb: float, limits: Dict) -> Tuple[int, str]:
    """
    Calculate how many splits are needed based on limits.
    
    Args:
        total_requests: Total number of requests
        estimated_size_mb: Estimated file size in MB
        limits: Provider limits
        
    Returns:
        Tuple of (splits_needed, reason)
    """
    request_splits = ceil(total_requests / limits['max_requests'])
    size_splits = ceil(estimated_size_mb / limits['max_file_size_mb'])
    
    splits_needed = max(request_splits, size_splits)
    
    reasons = []
    if request_splits > 1:
        reasons.append(f"requests ({total_requests:,} > {limits['max_requests']:,})")
    if size_splits > 1:
        reasons.append(f"file size ({estimated_size_mb:.1f}MB > {limits['max_file_size_mb']}MB)")
    
    reason = "due to " + " and ".join(reasons) if reasons else "file within limits"
    
    return splits_needed, reason

def create_batch_jsonl(model_name: str, prompt_file: str, examples_file, 
                      df: pd.DataFrame, output_path: str) -> List[str]:
    """
    Creates JSONL file(s) for batch processing with automatic splitting based on provider limits.
    
    Args:
        model_name (str): Name of the model to use (e.g., 'gpt-4o-mini', 'deepseek-r1-distill-llama-70b')
        prompt (str): System prompt for the model
        examples (list): List of example classifications
        df (pandas.DataFrame): DataFrame containing paragraphs to analyze
        output_path (str): Base path where the JSONL file(s) will be saved
        
    Returns:
        List[str]: List of paths to created JSONL files
    """
    # Detect provider and get limits
    provider = detect_provider_from_model(model_name)
    limits = get_batch_limits(provider)
    
    print(f"Detected provider: {limits['provider_name']}")
    print(f"Limits: {limits['max_requests']:,} requests, {limits['max_file_size_mb']}MB per file")
    
    examples = json.load(open(examples_file, 'r'))
    prompt = open(prompt_file, 'r').read()

    # Ensure examples have required fields
    for ex in examples:
        if 'reason' not in ex or not ex['reason'] or str(ex['reason']).lower() == 'nan':
            ex['reason'] = "Reasoning for the classification is not provided in the original example."
    
    # Build all JSONL lines first
    print("Building JSONL requests...")
    jsonl_lines = []
    for idx, row in df.iterrows():
        para_id = row['para_id']
        para_content = row['paragraphs']
        user_prompt = f"para_id: {para_id}\npara_content: {para_content}"
        
        # Add few-shot examples as messages
        few_shot_messages = []
        for ex in examples:
            few_shot_messages.append({
                "role": "user", 
                "content": f"para_id: {ex['para_id']}\npara_content: {ex['para_content']}"
            })
            
            # Safely handle the application_of_law_phrases
            try:
                application_of_law_phrases = ex.get('application_of_law_phrases', [])
                if isinstance(application_of_law_phrases, str):
                    application_of_law_phrases = application_of_law_phrases.replace("'", '"')
                    application_of_law_phrases = application_of_law_phrases.encode().decode('unicode_escape')
                
                application_of_law_phrases = json.loads(application_of_law_phrases) if isinstance(application_of_law_phrases, str) else application_of_law_phrases
            except json.JSONDecodeError:
                application_of_law_phrases = []
            
            few_shot_messages.append({
                "role": "assistant", 
                "content": json.dumps({
                    "para_id": ex['para_id'],
                    "if_law_applied": bool(ex['if_law_applied']),
                    "application_of_law_phrases": application_of_law_phrases,
                    "reason": ex['reason']
                })
            })
        
        jsonl_lines.append({
            "custom_id": f"request_{idx+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    *few_shot_messages,
                    {"role": "user", "content": user_prompt}
                ]
            }
        })
    
    # Check if splitting is needed
    total_requests = len(jsonl_lines)
    estimated_size_mb = estimate_file_size_mb(jsonl_lines)
    splits_needed, reason = calculate_splits_needed(total_requests, estimated_size_mb, limits)
    
    print(f"\nAnalysis:")
    print(f"  Total requests: {total_requests:,}")
    print(f"  Estimated size: {estimated_size_mb:.1f} MB")
    print(f"  Splits needed: {splits_needed} ({reason})")
    
    # Prepare output file paths
    base_path = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1] or '.jsonl'
    output_files = []
    
    if splits_needed == 1:
        # Single file - within limits
        print(f"\n✅ Creating single file: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for obj in jsonl_lines:
                f.write(json.dumps(obj) + '\n')
        
        # Verify final file
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Actual size: {final_size_mb:.1f} MB")
        print(f"   Requests: {total_requests:,}")
        
        output_files.append(output_path)
        
    else:
        # Multiple files needed
        print(f"\n📂 Creating {splits_needed} files...")
        requests_per_file = ceil(total_requests / splits_needed)
        
        for i in range(splits_needed):
            start_idx = i * requests_per_file
            end_idx = min((i + 1) * requests_per_file, total_requests)
            chunk = jsonl_lines[start_idx:end_idx]
            
            # Create filename for this chunk
            if splits_needed > 1:
                split_filename = f"{base_path}_part_{i+1:02d}{extension}"
            else:
                split_filename = output_path
            
            # Write chunk to file
            with open(split_filename, 'w', encoding='utf-8') as f:
                for obj in chunk:
                    f.write(json.dumps(obj) + '\n')
            
            # Verify chunk file
            chunk_size_mb = os.path.getsize(split_filename) / (1024 * 1024)
            chunk_requests = len(chunk)
            
            # Check compliance
            size_ok = chunk_size_mb <= limits['max_file_size_mb']
            requests_ok = chunk_requests <= limits['max_requests']
            
            status = "✅" if (size_ok and requests_ok) else "⚠️"
            print(f"   {status} {split_filename}")
            print(f"      Size: {chunk_size_mb:.1f} MB (limit: {limits['max_file_size_mb']} MB)")
            print(f"      Requests: {chunk_requests:,} (limit: {limits['max_requests']:,})")
            
            if not (size_ok and requests_ok):
                print(f"      ⚠️  WARNING: File may still exceed limits!")
            
            output_files.append(split_filename)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   Provider: {limits['provider_name']}")
    print(f"   Model: {model_name}")
    print(f"   Files created: {len(output_files)}")
    print(f"   Total requests: {total_requests:,}")
    
    total_size = sum(os.path.getsize(f) for f in output_files) / (1024 * 1024)
    print(f"   Total size: {total_size:.1f} MB")
    
    print(f"\n📁 Output files:")
    for i, file_path in enumerate(output_files, 1):
        print(f"   {i}. {file_path}")
    
    return output_files


# Example usage function for disagreement resolution
def create_disagreement_resolution_prompt() -> str:
    """
    Creates a sample prompt for disagreement resolution.
    Save this to a file and use with create_batch_jsonl_for_disagreement.
    """
    prompt = """You are analyzing paragraphs from UK case law to determine if they contain an application of law to specific facts.

APPLICATION OF LAW DEFINITION:
An application of law occurs when statutory legal provisions are applied directly to the specific facts of the case. This is distinct from merely discussing the law or citing statutes without linking them to the facts at hand.
IMPORTANT: Legal provisions can be applied BOTH explicitly (with direct statutory citation) AND implicitly (without naming specific statutes). Courts often apply legal principles without explicitly citing the statute number.

KEY INDICATORS OF APPLICATION OF LAW:
1. The judge connects specific statutory legal provisions to the specific factual circumstances
2. The text shows reasoning explaining how the law addresses or resolves the unique facts of the case
3. The paragraph includes judicial analysis that leads to a conclusion based on legal principles
4. Legal tests or criteria are applied to the facts of the case

NOT APPLICATION OF LAW:
1. Mere citations of statutes, cases, or legal principles without connecting them to the facts
2. Background procedural information or case history
3. Statements about jurisdiction or general legal explanations
4. Summaries of arguments made by parties without judicial analysis
5. Restatements of previous cases without connecting them to current facts

Your task is to resolve disagreements between two models by:
1. Carefully analyzing both models' reasoning
2. Evaluating which model's interpretation better aligns with the definition above
3. Providing your own classification with detailed justification
4. Indicating your confidence level and which model (if any) you agree with

Focus on evidence-based decision making and provide clear reasoning for your final classification."""
    
    return prompt

def create_batch_jsonl_for_disagreement(model_name: str, 
                                        df: pd.DataFrame, output_path: str) -> List[str]:
    """
    Creates JSONL file(s) for disagreement resolution using Claude as a judge.
    Handles cases where OpenAI and Llama models disagree on classification.
    
    Args:
        model_name (str): Name of the Claude model to use (e.g., 'claude-3-5-sonnet-20241022')
        prompt_file (str): Path to file containing the system prompt for disagreement resolution
        df (pandas.DataFrame): DataFrame with columns:
            - para_id: Paragraph identifier
            - paragraphs: Paragraph content
            - if_law_applied: OpenAI model decision (bool)
            - application_of_law_phrases: OpenAI model phrases (list/str)
            - reason: OpenAI model reasoning (str)
            - if_law_applied_llama: Llama model decision (bool)
            - application_of_law_phrases_llama: Llama model phrases (list/str)
            - reason_llama: Llama model reasoning (str)
        output_path (str): Base path where the JSONL file(s) will be saved
        
    Returns:
        List[str]: List of paths to created JSONL files
    """
    # Detect provider and get limits (Claude is typically through Anthropic API)
    provider = 'anthropic'  # Assuming Claude is accessed through Anthropic API
    # Use OpenAI limits as fallback since Anthropic has similar constraints
    limits = get_batch_limits('anthropic')
    limits['provider_name'] = 'Anthropic (Claude)'
    
    print(f"Using provider: {limits['provider_name']}")
    print(f"Limits: {limits['max_requests']:,} requests, {limits['max_file_size_mb']}MB per file")
    
    # Load the disagreement resolution prompt
    prompt = create_disagreement_resolution_prompt()
    
    # Filter for disagreement cases only
    disagreement_df = df[df['if_law_applied'] != df['if_law_applied_llama']].copy()
    
    if disagreement_df.empty:
        print("No disagreement cases found in the dataset!")
        return []
    
    print(f"Found {len(disagreement_df)} disagreement cases out of {len(df)} total cases ({len(disagreement_df)/len(df)*100:.1f}%)")
    
    # Build all JSONL lines for disagreement resolution
    print("Building disagreement resolution JSONL requests...")
    jsonl_lines = []
    
    for idx, row in disagreement_df.iterrows():
        para_id = row['para_id']
        para_content = row['paragraphs']
        
        # OpenAI model results
        openai_decision = bool(row['if_law_applied'])
        openai_phrases = row['application_of_law_phrases']
        openai_reason = row['reason']
        
        # Llama model results  
        llama_decision = bool(row['if_law_applied_llama'])
        llama_phrases = row['application_of_law_phrases_llama']
        llama_reason = row['reason_llama']
        
        # Safely handle phrase lists for both models
        def safe_parse_phrases(phrases):
            if pd.isna(phrases) or phrases == '':
                return []
            if isinstance(phrases, str):
                try:
                    # Handle string representations of lists
                    phrases_clean = phrases.replace("'", '"').encode().decode('unicode_escape')
                    return json.loads(phrases_clean) if phrases_clean.startswith('[') else [phrases_clean]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return [str(phrases)]
            elif isinstance(phrases, list):
                return phrases
            else:
                return [str(phrases)]
        
        openai_phrases_clean = safe_parse_phrases(openai_phrases)
        llama_phrases_clean = safe_parse_phrases(llama_phrases)
        
        # Handle NaN/missing reasons
        openai_reason = str(openai_reason) if not pd.isna(openai_reason) else "No reasoning provided"
        llama_reason = str(llama_reason) if not pd.isna(llama_reason) else "No reasoning provided"
        
        # Create the disagreement resolution prompt
        user_prompt = f"""You are analyzing a legal paragraph to determine if it contains an application of law to specific facts.

Two models have disagreed on this classification:

**Model A (OpenAI GPT) Decision:** {openai_decision}
**Model A Reasoning:** {openai_reason}
**Model A Phrases:** {json.dumps(openai_phrases_clean)}

**Model B (Llama) Decision:** {llama_decision}  
**Model B Reasoning:** {llama_reason}
**Model B Phrases:** {json.dumps(llama_phrases_clean)}

**Paragraph ID:** {para_id}
**Paragraph Content:** {para_content}

Based on the evidence and reasoning from both models, provide your analysis in the following JSON format:
{{
    "para_id": "{para_id}",
    "if_law_applied": boolean,
    "application_of_law_phrases": ["list", "of", "phrases"],
    "reason": "Your detailed reasoning explaining which model's analysis is more accurate and why",
    "confidence": "High/Medium/Low",
    "agreement_with": "OpenAI/Llama - specify which model you agree with"
}}"""
        
        jsonl_lines.append({
            "custom_id": f"request_{idx+1}",
            "method": "POST", 
            "url": "/v1/chat/completions",  # Adjust based on actual Anthropic API endpoint
            "body": {
                "model": model_name,
                "max_tokens": 1000,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        })
    
    # Check if splitting is needed
    total_requests = len(jsonl_lines)
    estimated_size_mb = estimate_file_size_mb(jsonl_lines)
    splits_needed, reason = calculate_splits_needed(total_requests, estimated_size_mb, limits)
    
    print(f"\nAnalysis:")
    print(f"  Disagreement cases: {total_requests:,}")
    print(f"  Estimated size: {estimated_size_mb:.1f} MB")
    print(f"  Splits needed: {splits_needed} ({reason})")
    
    # Prepare output file paths
    base_path = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1] or '.jsonl'
    output_files = []
    
    if splits_needed == 1:
        # Single file - within limits
        print(f"\n✅ Creating single disagreement resolution file: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for obj in jsonl_lines:
                f.write(json.dumps(obj) + '\n')
        
        # Verify final file
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Actual size: {final_size_mb:.1f} MB")
        print(f"   Disagreement cases: {total_requests:,}")
        
        output_files.append(output_path)
        
    else:
        # Multiple files needed
        print(f"\n📂 Creating {splits_needed} disagreement resolution files...")
        requests_per_file = ceil(total_requests / splits_needed)
        
        for i in range(splits_needed):
            start_idx = i * requests_per_file
            end_idx = min((i + 1) * requests_per_file, total_requests)
            chunk = jsonl_lines[start_idx:end_idx]
            
            # Create filename for this chunk
            if splits_needed > 1:
                split_filename = f"{base_path}_disagreement_part_{i+1:02d}{extension}"
            else:
                split_filename = output_path
            
            # Write chunk to file
            with open(split_filename, 'w', encoding='utf-8') as f:
                for obj in chunk:
                    f.write(json.dumps(obj) + '\n')
            
            # Verify chunk file
            chunk_size_mb = os.path.getsize(split_filename) / (1024 * 1024)
            chunk_requests = len(chunk)
            
            # Check compliance
            size_ok = chunk_size_mb <= limits['max_file_size_mb']
            requests_ok = chunk_requests <= limits['max_requests']
            
            status = "✅" if (size_ok and requests_ok) else "⚠️"
            print(f"   {status} {split_filename}")
            print(f"      Size: {chunk_size_mb:.1f} MB (limit: {limits['max_file_size_mb']} MB)")
            print(f"      Requests: {chunk_requests:,} (limit: {limits['max_requests']:,})")
            
            if not (size_ok and requests_ok):
                print(f"      ⚠️  WARNING: File may still exceed limits!")
            
            output_files.append(split_filename)
    
    # Summary
    print(f"\n📋 Disagreement Resolution Summary:")
    print(f"   Provider: {limits['provider_name']}")
    print(f"   Model: {model_name}")
    print(f"   Files created: {len(output_files)}")
    print(f"   Disagreement cases processed: {total_requests:,}")
    print(f"   Agreement rate: {((len(df) - total_requests) / len(df) * 100):.1f}%")
    
    total_size = sum(os.path.getsize(f) for f in output_files) / (1024 * 1024)
    print(f"   Total size: {total_size:.1f} MB")
    
    print(f"\n📁 Output files:")
    for i, file_path in enumerate(output_files, 1):
        print(f"   {i}. {file_path}")
    
    return output_files


def create_batch_jsonl_for_phrase_extraction(model_name: str, prompt_file: str, examples_file, 
                                           df: pd.DataFrame, output_path: str) -> List[str]:
    """
    Creates JSONL file(s) for phrase extraction with legislation section text.
    This function is specifically for extracting exact phrases that match between case law and legislation.
    
    Args:
        model_name (str): Name of the model to use (e.g., 'gpt-4o-mini', 'deepseek-r1-distill-llama-70b')
        prompt_file (str): Path to file containing the system prompt for phrase extraction
        examples_file: Path to file containing example extractions
        df (pandas.DataFrame): DataFrame containing paragraphs and section_text to analyze
        output_path (str): Base path where the JSONL file(s) will be saved
        
    Returns:
        List[str]: List of paths to created JSONL files
    """
    # Detect provider and get limits
    provider = detect_provider_from_model(model_name)
    limits = get_batch_limits(provider)
    
    print(f"Detected provider: {limits['provider_name']}")
    print(f"Limits: {limits['max_requests']:,} requests, {limits['max_file_size_mb']}MB per file")
    
    examples = json.load(open(examples_file, 'r'))
    prompt = open(prompt_file, 'r').read()

    # Ensure examples have required fields
    for ex in examples:
        if 'reason' not in ex or not ex['reason'] or str(ex['reason']).lower() == 'nan':
            ex['reason'] = "Reasoning for the extraction is not provided in the original example."
    
    # Build all JSONL lines first
    print("Building phrase extraction JSONL requests...")
    jsonl_lines = []
    for idx, row in df.iterrows():
        para_id = row['para_id']
        para_content = row['paragraphs']
        section_text = row.get('section_text', '')
        section_id = row.get('section_id', '')
        
        # Create user prompt with paragraph, legislation text, and section_id
        user_prompt = f"para_id: {para_id}\npara_content: {para_content}\nsection_text: {section_text}\nsection_id: {section_id}"
        
        # Add few-shot examples as messages
        few_shot_messages = []
        for ex in examples:
            # Example user message with paragraph, legislation, and section_id
            ex_para_content = ex.get('para_content', '')
            ex_section_text = ex.get('section_text', '')
            ex_section_id = ex.get('section_id', '')
            few_shot_messages.append({
                "role": "user", 
                "content": f"para_id: {ex['para_id']}\npara_content: {ex_para_content}\nsection_text: {ex_section_text}\nsection_id: {ex_section_id}"
            })
            
            # Safely handle the extracted phrases
            try:
                extracted_phrases = ex.get('extracted_phrases', [])
                if isinstance(extracted_phrases, str):
                    extracted_phrases = extracted_phrases.replace("'", '"')
                    extracted_phrases = extracted_phrases.encode().decode('unicode_escape')
                
                extracted_phrases = json.loads(extracted_phrases) if isinstance(extracted_phrases, str) else extracted_phrases
            except json.JSONDecodeError:
                extracted_phrases = []
            
            few_shot_messages.append({
                "role": "assistant", 
                "content": json.dumps({
                    "para_id": ex['para_id'],
                    "section_id": ex_section_id,
                    "extracted_phrases": extracted_phrases
                })
            })
        
        jsonl_lines.append({
            "custom_id": f"request_{idx+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    *few_shot_messages,
                    {"role": "user", "content": user_prompt}
                ]
            }
        })
    
    # Check if splitting is needed
    total_requests = len(jsonl_lines)
    estimated_size_mb = estimate_file_size_mb(jsonl_lines)
    splits_needed, reason = calculate_splits_needed(total_requests, estimated_size_mb, limits)
    
    print(f"\nAnalysis:")
    print(f"  Total requests: {total_requests:,}")
    print(f"  Estimated size: {estimated_size_mb:.1f} MB")
    print(f"  Splits needed: {splits_needed} ({reason})")
    
    # Prepare output file paths
    base_path = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1] or '.jsonl'
    output_files = []
    
    if splits_needed == 1:
        # Single file - within limits
        print(f"\n✅ Creating single file: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for obj in jsonl_lines:
                f.write(json.dumps(obj) + '\n')
        
        # Verify final file
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Actual size: {final_size_mb:.1f} MB")
        print(f"   Requests: {total_requests:,}")
        
        output_files.append(output_path)
        
    else:
        # Multiple files needed
        print(f"\n📂 Creating {splits_needed} files...")
        requests_per_file = ceil(total_requests / splits_needed)
        
        for i in range(splits_needed):
            start_idx = i * requests_per_file
            end_idx = min((i + 1) * requests_per_file, total_requests)
            chunk = jsonl_lines[start_idx:end_idx]
            
            # Create filename for this chunk
            if splits_needed > 1:
                split_filename = f"{base_path}_part_{i+1:02d}{extension}"
            else:
                split_filename = output_path
            
            # Write chunk to file
            with open(split_filename, 'w', encoding='utf-8') as f:
                for obj in chunk:
                    f.write(json.dumps(obj) + '\n')
            
            # Verify chunk file
            chunk_size_mb = os.path.getsize(split_filename) / (1024 * 1024)
            chunk_requests = len(chunk)
            
            # Check compliance
            size_ok = chunk_size_mb <= limits['max_file_size_mb']
            requests_ok = chunk_requests <= limits['max_requests']
            
            status = "✅" if (size_ok and requests_ok) else "⚠️"
            print(f"   {status} {split_filename}")
            print(f"      Size: {chunk_size_mb:.1f} MB (limit: {limits['max_file_size_mb']} MB)")
            print(f"      Requests: {chunk_requests:,} (limit: {limits['max_requests']:,})")
            
            if not (size_ok and requests_ok):
                print(f"      ⚠️  WARNING: File may still exceed limits!")
            
            output_files.append(split_filename)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   Provider: {limits['provider_name']}")
    print(f"   Model: {model_name}")
    print(f"   Files created: {len(output_files)}")
    print(f"   Total requests: {total_requests:,}")
    
    total_size = sum(os.path.getsize(f) for f in output_files) / (1024 * 1024)
    print(f"   Total size: {total_size:.1f} MB")
    
    print(f"\n📁 Output files:")
    for i, file_path in enumerate(output_files, 1):
        print(f"   {i}. {file_path}")
    
    return output_files

