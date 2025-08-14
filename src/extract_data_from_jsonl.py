import json
import pandas as pd
from util import clean_dataframe_for_csv
import re 
import os
import json5
import demjson3
def clean_json_content(content):
    """
    Clean JSON content that might be wrapped in markdown code blocks or have other formatting issues.
    
    Args:
        content: Raw content string from Claude
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    if not content:
        return content
    
    # Strip whitespace
    cleaned = content.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    
    # Strip again after removing code blocks
    cleaned = cleaned.strip()
    
    # Handle case where content might have extra newlines or spaces
    # Try to find the JSON object boundaries
    if cleaned.startswith('{') and cleaned.endswith('}'):
        return cleaned
    
    # If it doesn't start/end with braces, try to find the JSON object
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return cleaned[start_idx:end_idx + 1]
    
    return cleaned

def extract_data_from_llama_jsonl_after_verification(input_jsonl_path, output_jsonl_path, output_csv_path):
    """
    Extract data from Llama JSONL files after verification.
    
    Args:
        input_jsonl_path: Path to the input JSONL file containing requests
        output_jsonl_path: Path to the output JSONL file containing responses
        output_csv_path: Path to save the cleaned CSV file
    """
    # Dictionary to store input data by custom_id
    input_data = {}
    
    # Read input JSONL file to extract para_id, paragraph_text, section_text, section_id
    print(f"Reading input file: {input_jsonl_path}")
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                
                # Extract user content from the request
                messages = obj.get('body', {}).get('messages', [])
                user_message = None
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '')
                        break
                
                if user_message:
                    # Parse the user content to extract para_id, para_content, section_text, section_id
                    lines = user_message.split('\n')
                    para_id = ''
                    para_content = ''
                    section_text = ''
                    section_id = ''
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('para_id:'):
                            para_id = line.replace('para_id:', '').strip()
                        elif line.startswith('para_content:'):
                            para_content = line.replace('para_content:', '').strip()
                        elif line.startswith('section_text:'):
                            section_text = line.replace('section_text:', '').strip()
                        elif line.startswith('section_id:'):
                            section_id = line.replace('section_id:', '').strip()
                    
                    input_data[custom_id] = {
                        'para_id': para_id,
                        'paragraph_text': para_content,
                        'section_text': section_text,
                        'section_id': section_id
                    }
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing input JSON: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing input line: {e}")
                with open('error_requests.txt', 'a', encoding='utf-8') as error_file:
                    error_file.write(f"{line.strip()}\n")
                continue
    
    
def extract_data_from_deepseek_jsonl_after_verification(input_jsonl_path, output_jsonl_path, output_csv_path):
    """
    Extract data from DeepSeek JSONL files after verification.
    
    Args:
        input_jsonl_path: Path to the input JSONL file containing requests
        output_jsonl_path: Path to the output JSONL file containing responses
        output_csv_path: Path to save the cleaned CSV file
    """
    # Dictionary to store input data by custom_id
    input_data = {}
    
    # Read input JSONL file to extract para_id, paragraph_text, section_text, section_id
    print(f"Reading input file: {input_jsonl_path}")
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                
                # Extract user content from the request
                messages = obj.get('body', {}).get('messages', [])
                user_message = None
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '')
                        break
                
                if user_message:
                    # Parse the user content to extract para_id, para_content, section_text, section_id
                    lines = user_message.split('\n')
                    para_id = ''
                    para_content = ''
                    section_text = ''
                    section_id = ''
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('para_id:'):
                            para_id = line.replace('para_id:', '').strip()
                        elif line.startswith('para_content:'):
                            para_content = line.replace('para_content:', '').strip()
                        elif line.startswith('section_text:'):
                            section_text = line.replace('section_text:', '').strip()
                        elif line.startswith('section_id:'):
                            section_id = line.replace('section_id:', '').strip()
                    
                    input_data[custom_id] = {
                        'para_id': para_id,
                        'paragraph_text': para_content,
                        'section_text': section_text,
                        'section_id': section_id
                    }
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing input JSON: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing input line: {e}")
                continue
    
    print(f"Loaded {len(input_data)} input records")
    
    # Process output JSONL file
    records = []
    failed_requests = []
    print(f"Reading output file: {output_jsonl_path}")
    
    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                
                # Get the corresponding input data
                input_record = input_data.get(custom_id, {})
                
                # Extract response content
                response = obj.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    
                    # Extract thinking and JSON parts
                    thinking = ''
                    json_content = ''
                    
                    # Look for <think> tags
                    think_start = content.find('<think>')
                    think_end = content.find('</think>')
                    
                    if think_start != -1 and think_end != -1:
                        thinking = content[think_start + 7:think_end].strip()
                        # Remove thinking part from content for JSON extraction
                        content = content[:think_start] + content[think_end + 8:]
                    
                    # Extract JSON from the remaining content
                    json_content = ''
                    
                    # First try to find ```json blocks
                    json_block_start = content.find('```json')
                    if json_block_start != -1:
                        # Find the end of the json block
                        json_block_end = content.find('```', json_block_start + 7)
                        if json_block_end != -1:
                            json_content = content[json_block_start + 7:json_block_end].strip()
                    else:
                        # If no ```json block found, look for regular ``` blocks
                        code_block_start = content.find('```')
                        if code_block_start != -1:
                            code_block_end = content.find('```', code_block_start + 3)
                            if code_block_end != -1:
                                json_content = content[code_block_start + 3:code_block_end].strip()
                        else:
                            # Fallback to curly brackets - handle nested brackets properly
                            json_start = content.find('{')
                            if json_start != -1:
                                # Count brackets to find the matching closing brace
                                brace_count = 0
                                json_end = -1
                                
                                for i in range(json_start, len(content)):
                                    if content[i] == '{':
                                        brace_count += 1
                                    elif content[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i
                                            break
                                
                                if json_end != -1:
                                    json_content = content[json_start:json_end + 1]
                    
                    if json_content:
                        
                        try:
                            # Parse the JSON response
                            parsed_json = json.loads(json_content)
                            
                            # Create record with all extracted data
                            record = {
                                'custom_id': custom_id,
                                'para_id': input_record.get('para_id', ''),
                                'paragraph_text': input_record.get('paragraph_text', ''),
                                'section_text': input_record.get('section_text', ''),
                                'section_id': input_record.get('section_id', ''),
                                'thinking': thinking.replace('\n', ' '),
                                'extracted_phrases': parsed_json.get('extracted_phrases', []),
                                'reason': parsed_json.get('reason', ''),
                                'llm_para_id': parsed_json.get('para_id', ''),
                                'llm_section_id': parsed_json.get('section_id', '')
                            }
                            
                            records.append(record)
                            
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON content for custom_id {custom_id}: {e}")
                            print(f"JSON content: {json_content[:200]}...")
                            # Add to failed requests
                            failed_requests.append({
                                'custom_id': custom_id,
                                'para_id': input_record.get('para_id', ''),
                                'error': f"JSON parsing error: {e}",
                                'content': content  # First 500 chars for debugging
                            })
                            continue
                    else:
                        print(f"No JSON content found for custom_id {custom_id}")
                        # Add to failed requests
                        failed_requests.append({
                            'custom_id': custom_id,
                            'para_id': input_record.get('para_id', ''),
                            'error': "No JSON content found",
                            'content': content  # First 500 chars for debugging
                        })
                        continue
                        
            except json.JSONDecodeError as e:
                print(f"Error parsing output JSON: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing output line: {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    if df.empty:
        print("Warning: No valid records found")
        return None
    
    # Clean up the DataFrame
    # Convert extracted_phrases list to string for CSV storage
    df['extracted_phrases'] = df['extracted_phrases'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    # Clean text columns to remove newlines and extra whitespace
    df = clean_dataframe_for_csv(df)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Clean CSV saved to: {output_csv_path}")
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Save failed requests to text file
    if failed_requests:
        failed_requests_path = output_csv_path.replace('.csv', '_failed_requests.txt')
        with open(failed_requests_path, 'w', encoding='utf-8') as f:
            for i, failed in enumerate(failed_requests):
                f.write(f"Failed Request #{i+1}\n")
                f.write(f"Custom ID: {failed['custom_id']}\n")
                f.write(f"Para ID: {failed['para_id']}\n")
                f.write(f"Error: {failed['error']}\n")
                f.write(f"Content: {failed['content']}\n")
                f.write("=" * 50 + "\n\n")
        
        print(f"❌ Failed requests saved to: {failed_requests_path}")
        print(f"📊 Total failed requests: {len(failed_requests)}")
    
    # Show sample data
    print("\n📋 Sample data:")
    print(df[['custom_id', 'para_id', 'section_id', 'reason']].head())
    
    return df

def extract_phrases_data_from_openai_jsonl(input_jsonl_path, output_jsonl_path, output_csv_path):
    """
    Extracts and merges phrase data from OpenAI JSONL input and output files, and saves to CSV.
    Ensures para_text and section_text are matched by custom_id.
    """
    # Step 1: Read input JSONL and build a dict by custom_id
    input_data = {}
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                messages = obj.get('body', {}).get('messages', [])
                user_message = None
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '')
                        break
                para_id = ''
                section_id = ''
                paragraph_text = ''
                section_text = ''
                for l in user_message.split('\n'):
                    l = l.strip()
                    if l.startswith('para_id:'):
                        para_id = l.replace('para_id:', '').strip()
                    elif l.startswith('section_id:'):
                        section_id = l.replace('section_id:', '').strip()
                    elif l.startswith('para_content:'):
                        paragraph_text = l.replace('para_content:', '').strip()
                    elif l.startswith('section_text:'):
                        section_text = l.replace('section_text:', '').strip()
                input_data[custom_id] = {
                    'para_id': para_id,
                    'section_id': section_id,
                    'paragraph_text': paragraph_text,
                    'section_text': section_text
                }
            except Exception:
                continue

    # Step 2: Read output JSONL and extract response data using robust JSON extraction
    output_data = {}
    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                response = obj.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                para_id = ''
                section_id = ''
                extracted_phrases = []
                reason = ''
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    # Try to extract JSON from content (same logic as deepseek/llama)
                    json_content = ''
                    json_block_start = content.find('```json')
                    if json_block_start != -1:
                        json_block_end = content.find('```', json_block_start + 7)
                        if json_block_end != -1:
                            json_content = content[json_block_start + 7:json_block_end].strip()
                    else:
                        code_block_start = content.find('```')
                        if code_block_start != -1:
                            code_block_end = content.find('```', code_block_start + 3)
                            if code_block_end != -1:
                                json_content = content[code_block_start + 3:code_block_end].strip()
                        else:
                            json_start = content.find('{')
                            if json_start != -1:
                                brace_count = 0
                                json_end = -1
                                for i in range(json_start, len(content)):
                                    if content[i] == '{':
                                        brace_count += 1
                                    elif content[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i
                                            break
                                if json_end != -1:
                                    json_content = content[json_start:json_end + 1]
                    if json_content:
                        try:
                            parsed_json = json.loads(json_content)
                            para_id = parsed_json.get('para_id', '')
                            section_id = parsed_json.get('section_id', '')
                            extracted_phrases = parsed_json.get('extracted_phrases', [])
                            reason = parsed_json.get('reason', '')
                        except Exception:
                            continue
                output_data[custom_id] = {
                    'para_id': para_id,
                    'section_id': section_id,
                    'extracted_phrases': extracted_phrases,
                    'reason': reason
                }
            except Exception:
                continue

    # Step 3: Merge input and output data by custom_id
    records = []
    for custom_id, inp in input_data.items():
        out = output_data.get(custom_id, {})
        records.append({
            'custom_id': custom_id,
            'para_id': inp.get('para_id', ''),
            'section_id': inp.get('section_id', ''),
            'paragraph_text': inp.get('paragraph_text', ''),
            'section_text': inp.get('section_text', ''),
            'extracted_phrases': json.dumps(out.get('extracted_phrases', [])),
            'reason': out.get('reason', '')
        })

    # Step 4: Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ OpenAI phrases CSV saved to: {output_csv_path}")
    print(df.head())

def merge_combined_csv_and_claude_jsonl(combined_csv_path, output_jsonl_path, output_csv_path):
    """
    Reads combined.csv and Claude output.jsonl, extracts 'para_id' and relevant response details from output.jsonl,
    merges them on 'para_id', and saves the result to output.csv.
    
    Args:
        combined_csv_path: Path to the combined.csv file
        output_jsonl_path: Path to the Claude output.jsonl file
        output_csv_path: Path to save the merged output.csv file
    """
    
    
    # Read combined.csv into DataFrame
    df_combined = pd.read_csv(combined_csv_path)
    
    # Read output.jsonl, extract para_id and the response, and create a DataFrame
    records = []
    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                
                # Extract content from Claude's response structure
                content = obj.get('result', {}).get('message', {}).get('content', [])
                
                # Claude returns content as a list of objects with 'type' and 'text'
                if content and isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', '')
                    
                    if text_content:
                        try:
                            # Clean the text content using the helper function
                            text_content = text_content.replace('```json', '')
                            text_content = text_content.replace('```', '')

                            cleaned_content = clean_json_content(text_content)
                            
                            # Parse the JSON content from Claude's response
                            parsed_content = json.loads(cleaned_content)
                            response_data = {
                                'para_id': parsed_content.get('para_id', None),
                                'if_law_applied': parsed_content.get('if_law_applied', False),
                                'application_of_law_phrases': parsed_content.get('application_of_law_phrases', []),
                                'reason': parsed_content.get('reason', ''),
                                'confidence': parsed_content.get('confidence', ''),  # Claude-specific field
                                'agreement_with': parsed_content.get('agreement_with', '')  # Claude-specific field
                            }
                            records.append(response_data)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing content JSON: {e}")
                            print(f"Original content: {text_content[:200]}...")  # Print first 200 chars for debugging
                            print(f"Cleaned content: {cleaned_content[:200]}...")  # Print cleaned content for debugging
                            continue
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line JSON: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing line: {e}")
                continue

    # Convert the records to DataFrame
    df_output = pd.DataFrame(records)
    
    if df_output.empty:
        print("Warning: No valid records found in the JSONL file")
        return

    # Check if 'para_id' exists in both DataFrames
    if 'para_id' not in df_combined.columns or 'para_id' not in df_output.columns:
        print("Combined CSV columns:", df_combined.columns.tolist())
        print("Output DataFrame columns:", df_output.columns.tolist())
        raise ValueError("para_id must be present in both files for merging.")
    
    print(f"Found {len(df_output)} records in JSONL file")
    print(f"Combined CSV has {len(df_combined)} records")

    # Merge the two DataFrames on 'para_id' (this will preserve all columns from combined.csv)
    merged_df = pd.merge(df_combined, df_output, on='para_id', how='left', suffixes=('', '_claude'))

    # Save the merged result to a new CSV
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged CSV saved to: {output_csv_path}")
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print("Sample of merged data:")
    print(merged_df[['para_id', 'if_law_applied', 'confidence', 'agreement_with']].head())
    
    return merged_df

def merge_combined_csv_and_output_jsonl(combined_csv_path, output_jsonl_path, output_csv_path):
    """
    Reads combined.csv and output.jsonl, extracts 'para_id' and relevant response details from output.jsonl,
    merges them on 'para_id', and saves the result to output.csv.
    """
    # Read combined.csv into DataFrame
    df_combined = pd.read_csv(combined_csv_path)
    
    # Read output.jsonl, extract para_id and the response, and create a DataFrame
    records = []
    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            # Extract para_id from the nested structure
            para_id = obj.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', None)
            
            if para_id:
                # Now parse the 'content' field which is a JSON string
                try:
                    parsed_content = json.loads(para_id)  # Parse content into a dictionary
                    response_data = {
                        'para_id': parsed_content.get('para_id', None),
                        'if_law_applied': parsed_content.get('if_law_applied', False),
                        'application_of_law_phrases': parsed_content.get('application_of_law_phrases', []),
                        'reason': parsed_content.get('reason', '')
                    }
                    records.append(response_data)
                except json.JSONDecodeError:
                    continue  # Skip if content cannot be decoded as JSON

    # Convert the records to DataFrame
    df_output = pd.DataFrame(records)

    # Check if 'para_id' exists in both DataFrames
    if 'para_id' not in df_combined.columns or 'para_id' not in df_output.columns:
        print(df_output.columns)
        print(df_combined.columns)
        raise ValueError("para_id must be present in both files for merging.")
    
    

    # Merge the two DataFrames on 'para_id' (this will preserve all columns from combined.csv)
    merged_df = pd.merge(df_combined, df_output, on='para_id', how='left')

    # Save the merged result to a new CSV
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged CSV saved to: {output_csv_path}")
    print(merged_df.head())
    print(merged_df.columns)

def extract_llama_jsonl_to_csv(input_jsonl_path,output_jsonl_path, output_csv_path):
    """
    Reads input and output JSONL files for Llama, merges them, and saves a CSV with required columns.
    """
    

    # Step 1: Read input JSONL and build a dict by custom_id
    input_data = {}
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                messages = obj.get('body', {}).get('messages', [])
                user_message = None
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '')
                        break
                para_id = ''
                para_content = ''
                section_text = ''
                for l in user_message.split('\n'):
                    l = l.strip()
                    if l.startswith('para_id:'):
                        para_id = l.replace('para_id:', '').strip()
                    elif l.startswith('para_content:'):
                        para_content = l.replace('para_content:', '').strip()
                    elif l.startswith('section_text:'):
                        section_text = l.replace('section_text:', '').strip()
                input_data[custom_id] = {
                    'para_id': para_id,
                    'paragraph_text': para_content,
                    'section_id': obj.get('section_id', ''),
                    'section_text': section_text
                }
            except Exception:
                continue

    # Step 2: Read output JSONL and extract response data using robust JSON extraction
    output_data = {}
    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                response = obj.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')

                    # Try to extract JSON from content (same logic as deepseek function)
                    json_content = ''
                    json_block_start = content.find('```json')
                    if json_block_start != -1:
                        json_block_end = content.find('```', json_block_start + 7)
                        if json_block_end != -1:
                            json_content = content[json_block_start + 7:json_block_end].strip()
                    else:
                        code_block_start = content.find('```')
                        if code_block_start != -1:
                            code_block_end = content.find('```', code_block_start + 3)
                            if code_block_end != -1:
                                json_content = content[code_block_start + 3:code_block_end].strip()
                        else:
                            json_start = content.find('{')
                            if json_start != -1:
                                brace_count = 0
                                json_end = -1
                                for i in range(json_start, len(content)):
                                    if content[i] == '{':
                                        brace_count += 1
                                    elif content[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i
                                            break
                                if json_end != -1:
                                    json_content = content[json_start:json_end + 1]
                    if json_content:
                        try:
                            parsed_json = json.loads(json_content)
                            output_data[custom_id] = {
                                'extracted_phrases': parsed_json.get('extracted_phrases', []),
                                'reason': parsed_json.get('reason', ''),
                                'para_id': parsed_json.get('para_id', ''),
                                'section_id': parsed_json.get('section_id', '')
                            }
                        except Exception:
                            print(f"Error parsing output JSON for custom_id {custom_id}: {e}")
                            continue
            except Exception as e:
                print(f"Unexpected error processing output line: {e}")
                continue

    # Step 3: Merge input and output data by custom_id
    records = []
    for custom_id, inp in input_data.items():
        out = output_data.get(custom_id, {})
        records.append({
            'custom_id': custom_id,
            'para_id': inp.get('para_id'),
            'section_text': inp.get('section_text'),
            'section_id': out.get('section_id'),
            'paragraph_text': inp.get('paragraph_text'),
            'extracted_phrases': json.dumps(out.get('extracted_phrases', [])),
            'reason': out.get('reason', '')
        })

    # Step 4: Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Llama CSV saved to: {output_csv_path}")
    print(df.head())

def extract_llama_of_few_shot_output(input_jsonl_file_path, output_jsonl_path, output_csv_path):
    """
    Extracts and merges phrase data from Llama few-shot output JSONL file, and saves to CSV.
    
    Args:
        input_jsonl_file_path: Path to the input JSONL file (for reference/matching)
        output_jsonl_path: Path to the Llama output JSONL file
        output_csv_path: Path to save the merged output CSV file
    """
    records = []
    
    # Load input data for reference (if needed for matching)
    input_data = {}
    try:
        with open(input_jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    input_obj = json.loads(line)
                    custom_id = input_obj.get('custom_id', '')
                    input_data[custom_id] = input_obj
    except FileNotFoundError:
        print(f"Warning: Input file {input_jsonl_file_path} not found. Proceeding without input data.")
    
    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                response = obj.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    
                    # Extract JSON from the content
                    json_content = ''
                    content_stripped = content.strip()
                    
                    # Method 1: If the entire content is JSON (starts and ends with braces)
                    if content_stripped.startswith('{') and content_stripped.endswith('}'):
                        json_content = content_stripped
                    
                    # Method 2: Look for ```json blocks
                    elif '```json' in content:
                        json_block_start = content.find('```json')
                        json_block_end = content.find('```', json_block_start + 7)
                        if json_block_end != -1:
                            json_content = content[json_block_start + 7:json_block_end].strip()
                    
                    # Method 3: Look for generic ``` blocks
                    elif '```' in content:
                        code_block_start = content.find('```')
                        code_block_end = content.find('```', code_block_start + 3)
                        if code_block_end != -1:
                            json_content = content[code_block_start + 3:code_block_end].strip()
                    
                    # Method 4: Extract JSON object by brace matching (fallback)
                    else:
                        json_start = content.find('{')
                        if json_start != -1:
                            brace_count = 0
                            json_end = -1
                            in_string = False
                            escape_next = False
                            
                            for i in range(json_start, len(content)):
                                char = content[i]
                                
                                if escape_next:
                                    escape_next = False
                                    continue
                                    
                                if char == '\\':
                                    escape_next = True
                                    continue
                                    
                                if char == '"' and not escape_next:
                                    in_string = not in_string
                                    continue
                                    
                                if not in_string:
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i + 1
                                            break
                            
                            if json_end != -1:
                                json_content = content[json_start:json_end].strip()
                    
                    # Parse the extracted JSON
                    if json_content:
                        try:
                            parsed_json = json.loads(json_content)
                            para_id = parsed_json.get('para_id', '')
                            section_id = parsed_json.get('section_id', '')
                            extracted_phrases = parsed_json.get('extracted_phrases', [])
                            
                            # Process each extracted phrase
                            for phrase_idx, phrase in enumerate(extracted_phrases):
                                record = {
                                    'custom_id': custom_id,
                                    'para_id': para_id,
                                    'section_id': section_id,
                                    'phrase_index': phrase_idx,
                                    'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                                    'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                                    'confidence': phrase.get('confidence', ''),
                                    'reasoning': phrase.get('reasoning', ''),
                                    'line_number': line_num,
                                    'raw_content': content[:200] + '...' if len(content) > 200 else content  # First 200 chars for debugging
                                }
                                records.append(record)
                            
                            # If no extracted phrases, still create a record to track the response
                            if not extracted_phrases:
                                record = {
                                    'custom_id': custom_id,
                                    'para_id': para_id,
                                    'section_id': section_id,
                                    'phrase_index': -1,
                                    'case_law_excerpt': '',
                                    'legislation_excerpt': '',
                                    'confidence': '',
                                    'reasoning': 'No phrases extracted',
                                    'line_number': line_num,
                                    'raw_content': content[:200] + '...' if len(content) > 200 else content
                                }
                                records.append(record)
                                
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing error in line {line_num} (custom_id: {custom_id}): {e}")
                            # Create error record
                            record = {
                                'custom_id': custom_id,
                                'para_id': 'ERROR',
                                'section_id': 'ERROR',
                                'phrase_index': -1,
                                'case_law_excerpt': '',
                                'legislation_excerpt': '',
                                'confidence': '',
                                'reasoning': f'JSON parsing error: {str(e)}',
                                'line_number': line_num,
                                'raw_content': json_content[:200] + '...' if len(json_content) > 200 else json_content
                            }
                            records.append(record)
                    else:
                        print(f"No JSON content found in line {line_num} (custom_id: {custom_id})")
                        # Create no-content record
                        record = {
                            'custom_id': custom_id,
                            'para_id': 'NO_JSON',
                            'section_id': 'NO_JSON',
                            'phrase_index': -1,
                            'case_law_excerpt': '',
                            'legislation_excerpt': '',
                            'confidence': '',
                            'reasoning': 'No JSON content found',
                            'line_number': line_num,
                            'raw_content': content[:200] + '...' if len(content) > 200 else content
                        }
                        records.append(record)
                        
            except json.JSONDecodeError as e:
                print(f"Line parsing error in line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error in line {line_num}: {e}")
                continue
    
    # Convert to DataFrame and save to CSV
    if records:
        df = pd.DataFrame(records)
        
        # Sort by custom_id and phrase_index for better organization
        df = df.sort_values(['custom_id', 'phrase_index'])
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Successfully extracted {len(records)} records to {output_csv_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total records: {len(records)}")
        print(f"Unique custom_ids: {df['custom_id'].nunique()}")
        print(f"Records with phrases: {len(df[df['phrase_index'] >= 0])}")
        print(f"Error records: {len(df[df['para_id'].isin(['ERROR', 'NO_JSON'])])}")
        
        # Show confidence distribution
        confidence_counts = df['confidence'].value_counts()
        if not confidence_counts.empty:
            print(f"\nConfidence distribution:")
            for conf, count in confidence_counts.items():
                print(f"  {conf}: {count}")
        
        return df
    else:
        print("No records extracted.")
        return pd.DataFrame()
import json
import csv
import pandas as pd


def extract_llama_output_to_csv(output_jsonl_path, output_csv_path,source_csv_path):
    """
    Extracts para_id, section_id, and extracted_phrases from LLaMA output JSONL
    and saves the data into a CSV.

    Args:
        output_jsonl_path: Path to the LLaMA output JSONL file
        output_csv_path: Path to save the extracted data as CSV

    Returns:
        DataFrame with extracted records
    """
    try:
        source_df = pd.read_csv(source_csv_path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to load source CSV: {e}")

    source_df = source_df.fillna('')  # Ensure no NaNs
    para_dict = source_df.set_index('para_id')['paragraphs'].to_dict()
    section_dict = source_df.set_index('section_id')['section_text'].to_dict()

    records = []

    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                content = obj.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                if not content:
                    print(f"No content found for custom_id: {custom_id} (line {line_num})")
                    continue

                # Try decoding content as JSON (supports double-encoded strings)
                parsed_json = None
                try:
                    first_pass = json.loads(content)
                    if isinstance(first_pass, str):
                        parsed_json = json.loads(first_pass)
                    elif isinstance(first_pass, dict):
                        parsed_json = first_pass
                except json.JSONDecodeError:
                    pass

                # Fallback to raw extraction if parsing fails
                if not parsed_json:
                    json_content = extract_json_from_content(content)
                    if not json_content:
                        print(f"No JSON found for custom_id: {custom_id} (line {line_num})")
                        records.append({
                            'custom_id': custom_id,
                            'para_id': 'NO_JSON',
                            'section_id': 'NO_JSON',
                            'para_text': 'NO_JSON',
                            'phrase_index': -1,
                            'case_law_excerpt': '',
                            'legislation_excerpt': '',
                            'confidence': '',
                            'reasoning': 'No JSON content extracted',
                            'line_number': line_num,
                            'status': 'ERROR'
                        })
                        continue
                    parsed_json = json.loads(json_content)

                para_id = parsed_json.get('para_id', '')
                section_id = parsed_json.get('section_id', '')
                para_text = para_dict.get(para_id, '')
                section_text = section_dict.get(section_id, '')
                extracted_phrases = parsed_json.get('extracted_phrases', [])

                if extracted_phrases:
                    for phrase_idx, phrase in enumerate(extracted_phrases):
                        records.append({
                            'custom_id': custom_id,
                            'para_id': para_id,
                            'section_id': section_id,
                            'para_text': para_text,
                            'section_text': section_text,
                            'phrase_index': phrase_idx,
                            'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                            'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                            'confidence': phrase.get('confidence', ''),
                            'reasoning': phrase.get('reasoning', ''),
                            'line_number': line_num,
                            'status': 'SUCCESS'
                        })
                else:
                    records.append({
                        'custom_id': custom_id,
                        'para_id': para_id,
                        'section_id': section_id,
                        'para_text': para_text,
                        'section_text': section_text,
                        'phrase_index': -1,
                        'case_law_excerpt': '',
                        'legislation_excerpt': '',
                        'confidence': '',
                        'reasoning': 'No phrases in extracted_phrases array',
                        'line_number': line_num,
                        'status': 'NO_PHRASES'
                    })

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Save to CSV
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(['custom_id', 'phrase_index'])
        df.to_csv(output_csv_path, index=False, encoding='utf-8')

        # Print summary
        print(f"\nExtraction Summary:")
        print(f"Total records: {len(records)}")
        print(f"Successful extractions: {len(df[df['status'] == 'SUCCESS'])}")
        print(f"Records with no phrases: {len(df[df['status'] == 'NO_PHRASES'])}")
        print(f"No JSON found: {len(df[df['status'] == 'ERROR'])}")

        success_df = df[df['status'] == 'SUCCESS']
        if not success_df.empty:
            print(f"\nConfidence distribution:")
            for conf, count in success_df['confidence'].value_counts().items():
                print(f"  {conf}: {count}")

        print(f"\nSaved to: {output_csv_path}")
        return df
    else:
        print("No records extracted.")
        return pd.DataFrame()


def extract_json_from_content(content):
    """
    Attempts to extract JSON from various text formats.
    """
    content = content.strip()

    # Strategy 1: Double decode if needed
    try:
        first_pass = json.loads(content)
        if isinstance(first_pass, str):
            return json.loads(first_pass)
        return first_pass
    except Exception as e:
        print(f"First pass JSON decode failed: {e}")
        print(first_pass)
        pass

    # Strategy 2: Extract from ```json blocks
    if '```json' in content:
        start = content.find('```json') + 7
        end = content.find('```', start)
        if end != -1:
            json_candidate = content[start:end].strip()
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 3: Extract from generic ``` blocks
    if '```' in content:
        start = content.find('```') + 3
        end = content.find('```', start)
        if end != -1:
            json_candidate = content[start:end].strip()
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Brace matching
    json_start = content.find('{')
    if json_start != -1:
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(content)):
            char = content[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_candidate = content[json_start:i + 1]
                        try:
                            return json.loads(json_candidate)
                        except json.JSONDecodeError:
                            break

    # Strategy 5: Regex fallback
    try:
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        for match in matches:
            if 'para_id' in match and 'section_id' in match:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Regex extraction failed: {e}")

    return None

import json
import re
import pandas as pd


def extract_openai_output_to_csv(output_jsonl_path, output_csv_path, source_csv_path,error_request):
    """
    Extracts para_id, section_id, and extracted_phrases from OpenAI-style output JSONL,
    looks up corresponding para_text and section_text from source CSV,
    and saves the enriched data into a CSV.

    Args:
        output_jsonl_path: Path to the OpenAI output JSONL file
        output_csv_path: Path to save the extracted data as CSV
        source_csv_path: Path to source CSV with para_id and section_id texts

    Returns:
        DataFrame with extracted records
    """
    # Load source CSV for text lookup
    try:
        source_df = pd.read_csv(source_csv_path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to load source CSV: {e}")

    source_df = source_df.fillna('')  # Ensure no NaNs
    para_dict = source_df.set_index('para_id')['paragraphs'].to_dict()
    section_dict = source_df.set_index('section_id')['section_text'].to_dict()

    records = []

    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', '')
                content = obj.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                if not content:
                    print(f"No content found for custom_id: {custom_id} (line {line_num})")
                    continue

                # Try standard JSON parsing
                parsed_json = None
                try:
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, str):
                        parsed_json = json.loads(parsed_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse content for custom_id {custom_id} (line {line_num}): {e}")
                    with open(error_request, 'a', encoding='utf-8') as error_file:
                        error_file.write(f"{line.strip()}\n")

                    # Fallback: regex extraction
                    para_id_match = re.search(r'"para_id"\s*:\s*"([^"]+)"', content)
                    section_id_match = re.search(r'"section_id"\s*:\s*"([^"]+)"', content)
                    phrase_blocks = re.findall(r'\{[^{}]*"case_law_excerpt"[^{}]*\}', content)

                    para_id = para_id_match.group(1) if para_id_match else 'NO_PARA_ID'
                    section_id = section_id_match.group(1) if section_id_match else 'NO_SECTION_ID'
                    para_text = para_dict.get(para_id, '')
                    section_text = section_dict.get(section_id, '')

                    if phrase_blocks:
                        for idx, phrase_block in enumerate(phrase_blocks):
                            case_law_excerpt = re.search(r'"case_law_excerpt"\s*:\s*"([^"]*)"', phrase_block)
                            legislation_excerpt = re.search(r'"legislation_excerpt"\s*:\s*"([^"]*)"', phrase_block)
                            confidence = re.search(r'"confidence"\s*:\s*"([^"]*)"', phrase_block)
                            reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)"', phrase_block)

                            records.append({
                                'custom_id': custom_id,
                                'para_id': para_id,
                                'section_id': section_id,
                                'phrase_index': idx,
                                'case_law_excerpt': case_law_excerpt.group(1) if case_law_excerpt else '',
                                'legislation_excerpt': legislation_excerpt.group(1) if legislation_excerpt else '',
                                'confidence': confidence.group(1) if confidence else '',
                                'reasoning': reasoning.group(1) if reasoning else '',
                                'para_text': para_text,
                                'section_text': section_text,
                                'line_number': line_num,
                                'status': 'REGEX_RECOVERY'
                            })
                    else:
                        records.append({
                            'custom_id': custom_id,
                            'para_id': para_id,
                            'section_id': section_id,
                            'phrase_index': -1,
                            'case_law_excerpt': '',
                            'legislation_excerpt': '',
                            'confidence': '',
                            'reasoning': 'Regex fallback: no phrases found',
                            'para_text': para_text,
                            'section_text': section_text,
                            'line_number': line_num,
                            'status': 'REGEX_NO_PHRASES'
                        })
                    continue

                # Normal case
                para_id = parsed_json.get('para_id', '')
                section_id = parsed_json.get('section_id', '')
                para_text = para_dict.get(para_id, '')
                section_text = section_dict.get(section_id, '')
                extracted_phrases = parsed_json.get('extracted_phrases', [])

                if extracted_phrases:
                    for phrase_idx, phrase in enumerate(extracted_phrases):
                        records.append({
                            'custom_id': custom_id,
                            'para_id': para_id,
                            'section_id': section_id,
                            'phrase_index': phrase_idx,
                            'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                            'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                            'confidence': phrase.get('confidence', ''),
                            'reasoning': phrase.get('reasoning', ''),
                            'para_text': para_text,
                            'section_text': section_text,
                            'line_number': line_num,
                            'status': 'SUCCESS'
                        })
                else:
                    records.append({
                        'custom_id': custom_id,
                        'para_id': para_id,
                        'section_id': section_id,
                        'phrase_index': -1,
                        'case_law_excerpt': '',
                        'legislation_excerpt': '',
                        'confidence': '',
                        'reasoning': 'No extracted_phrases found',
                        'para_text': para_text,
                        'section_text': section_text,
                        'line_number': line_num,
                        'status': 'NO_PHRASES'
                    })

            except Exception as e:
                print(f"Unexpected error at line {line_num}: {e}")
                continue

    # Save and return
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(['custom_id', 'phrase_index'])
        df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"\nExtraction Summary:")
        print(f"Total records: {len(records)}")
        print(f"Successful: {len(df[df['status'] == 'SUCCESS'])}")
        print(f"Regex recovered: {len(df[df['status'] == 'REGEX_RECOVERY'])}")
        print(f"No phrases: {len(df[df['status'].isin(['NO_PHRASES', 'REGEX_NO_PHRASES'])])}")
        print(f"Saved to: {output_csv_path}")

        return df
    else:
        print("No records extracted.")
        return pd.DataFrame()
def extract_universal_output_to_csv2(output_jsonl_path, output_csv_path, source_csv_path):
    """
    Universal extraction function that handles multiple response formats with robust parsing.
    Ensures exactly one record per JSONL line (one per custom_id).
    
    Args:
        output_jsonl_path: Path to the output JSONL file
        output_csv_path: Path to save the extracted data as CSV
        source_csv_path: Path to source CSV with para_text and section_text
        
    Returns:
        DataFrame with exactly the number of records as lines in the JSONL file
    """
    import re
    import json
    import pandas as pd
    
    # Load source CSV for text lookup
    try:
        source_df = pd.read_csv(source_csv_path, dtype=str)
        source_df = source_df.fillna('')
        
        # Dynamically find para_text and section_text columns
        para_col = None
        section_col = None
        
        for col in source_df.columns:
            if 'para' in col.lower() and 'text' in col.lower():
                para_col = col
            elif 'section' in col.lower() and 'text' in col.lower():
                section_col = col
        
        if not para_col:
            for col in source_df.columns:
                if 'para' in col.lower() and ('content' in col.lower() or 'paragraph' in col.lower()):
                    para_col = col
                    break
        
        if not section_col:
            for col in source_df.columns:
                if 'section' in col.lower() and 'content' in col.lower():
                    section_col = col
                    break
        
        print(f"Using columns: para_text='{para_col}', section_text='{section_col}'")
        
        # Create lookup dictionaries
        para_dict = {}
        section_dict = {}
        
        if para_col and 'para_id' in source_df.columns:
            para_dict = dict(zip(source_df['para_id'], source_df[para_col]))
        
        if section_col and 'section_id' in source_df.columns:
            section_dict = dict(zip(source_df['section_id'], source_df[section_col]))
            
    except Exception as e:
        print(f"Warning: Failed to load source CSV: {e}")
        para_dict = {}
        section_dict = {}

    def extract_content_from_response(obj):
        """Extract content from various response formats"""
        content = ''
        
        try:
            # Most common format: response.body.choices[0].message.content
            if 'response' in obj and 'body' in obj['response']:
                choices = obj['response']['body'].get('choices', [])
                if choices and len(choices) > 0:
                    content = choices[0].get('message', {}).get('content', '')
                    if content:
                        return content
            
            # Alternative format: response.choices[0].message.content
            if 'response' in obj and 'choices' in obj['response']:
                choices = obj['response']['choices']
                if choices and len(choices) > 0:
                    content = choices[0].get('message', {}).get('content', '')
                    if content:
                        return content
            
            if 'result' in obj and 'message' in obj['result']:
                content = obj['result']['message'].get('content', '')
                if content and len(content) > 0:
                    print(f"Extracted content from result.message: {content}")  # Debugging
                    print(f"Content type: {type(content)}")  # Debugging
                    response_content = content[0].get('text', {})
                    return response_content
            # Direct choices format
            if 'choices' in obj:
                choices = obj['choices']
                if choices and len(choices) > 0:
                    content = choices[0].get('message', {}).get('content', '')
                    if content:
                        return content
            
            # Simple content formats
            if 'response' in obj:
                if 'content' in obj['response']:
                    return obj['response']['content']
                
            if 'content' in obj:
                return obj['content']
                
            if 'message' in obj and 'content' in obj['message']:
                return obj['message']['content']
                
        except (KeyError, IndexError, TypeError) as e:
            pass
        
        return content

    def robust_json_extraction(content):
        """Extract JSON from content using multiple methods"""
        if not content or not isinstance(content, str):
            return None
            
        # Method 1: Clean and try direct parsing
        try:
            # Remove think blocks
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove markdown
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            cleaned = re.sub(r'^```\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Try direct parse
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except:
            pass
        
        # Method 2: Find JSON patterns
        json_patterns = [
            r'\{[^{}]*"para_id"[^{}]*"section_id"[^{}]*"extracted_phrases"[^{}]*\}',
            r'\{[^{}]*"extracted_phrases"[^{}]*"para_id"[^{}]*"section_id"[^{}]*\}',
            r'\{.*?"para_id".*?\}',
            r'\{.*?\}(?=\s*$)',
            r'\{.*?\}',
        ]
        
        for pattern in json_patterns:
            try:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in reversed(matches):  # Try last match first (usually the final JSON)
                    try:
                        result = json.loads(match)
                        if isinstance(result, dict) and ('para_id' in result or 'section_id' in result):
                            return result
                    except:
                        continue
            except:
                continue
        
        # Method 3: Extract JSON with bracket counting
        try:
            start_idx = content.find('{')
            if start_idx != -1:
                bracket_count = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(content[start_idx:], start_idx):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_str = content[start_idx:i+1]
                                try:
                                    result = json.loads(json_str)
                                    if isinstance(result, dict):
                                        return result
                                except:
                                    pass
                                break
        except:
            pass
        
        # Method 4: Try to fix common JSON issues
        try:
            # Get content after last opening brace
            last_brace = content.rfind('{')
            if last_brace != -1:
                json_candidate = content[last_brace:]
                
                # Fix common issues
                json_candidate = re.sub(r',(\s*[}\]])', r'\1', json_candidate)  # Remove trailing commas
                json_candidate = re.sub(r'([}\]]),\s*$', r'\1', json_candidate)  # Remove final trailing comma
                
                try:
                    result = json.loads(json_candidate)
                    if isinstance(result, dict):
                        return result
                except:
                    pass
        except:
            pass
        
        return None

    records = []
    total_lines = 0
    successful_lines = 0
    failed_lines = 0
    failed_request_ids = []

    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Default record for this line
            record = {
                'custom_id': f'UNKNOWN_{line_num}',
                'para_id': '',
                'section_id': '',
                'phrase_index': -1,
                'case_law_excerpt': '',
                'legislation_excerpt': '',
                'confidence': '',
                'reasoning': 'Processing failed',
                'para_text': '',
                'section_text': '',
                'line_number': line_num,
                'status': 'PROCESSING_FAILED'
            }
            
            if not line.strip():
                record.update({
                    'custom_id': f'EMPTY_LINE_{line_num}',
                    'reasoning': 'Empty line',
                    'status': 'EMPTY_LINE'
                })
                records.append(record)
                failed_lines += 1
                failed_request_ids.append(f'EMPTY_LINE_{line_num}')
                continue

            # Parse line JSON
            try:
                line_obj = json.loads(line)
                custom_id = line_obj.get('custom_id', f'UNKNOWN_{line_num}')
                record['custom_id'] = custom_id
            except json.JSONDecodeError as e:
                record.update({
                    'reasoning': f'Invalid JSONL line: {str(e)[:100]}',
                    'status': 'LINE_PARSE_ERROR'
                })
                records.append(record)
                failed_lines += 1
                failed_request_ids.append(record['custom_id'])
                continue
            
            # Extract content
            content = extract_content_from_response(line_obj)
            if not content:
                record.update({
                    'reasoning': 'No content found in response',
                    'status': 'NO_CONTENT'
                })
                records.append(record)
                failed_lines += 1
                failed_request_ids.append(custom_id)
                continue
            
            # Parse JSON content
            parsed_json = robust_json_extraction(content)
            if not parsed_json:
                record.update({
                    'reasoning': f'Failed to extract JSON from content. Preview: {content[:100]}...',
                    'status': 'JSON_EXTRACTION_FAILED'
                })
                records.append(record)
                failed_lines += 1
                failed_request_ids.append(custom_id)
                continue
            
            # Extract data from parsed JSON
            try:
                para_id = str(parsed_json.get('para_id', ''))
                section_id = str(parsed_json.get('section_id', ''))
                para_text = para_dict.get(para_id, '')
                section_text = section_dict.get(section_id, '')
                extracted_phrases = parsed_json.get('extracted_phrases', [])
                
                # Update record with extracted data
                record.update({
                    'para_id': para_id,
                    'section_id': section_id,
                    'para_text': para_text,
                    'section_text': section_text
                })
                
                # Handle extracted phrases
                if extracted_phrases and len(extracted_phrases) > 0:
                    for phrase_idx, phrase in enumerate(extracted_phrases):
                        
                        
                        if isinstance(phrase, dict):
                            case_law_excerpt = str(phrase.get('case_law_excerpt', ''))
                            legislation_excerpt = str(phrase.get('legislation_excerpt', ''))
                            confidence = str(phrase.get('confidence', ''))
                            reasoning = str(phrase.get('reasoning', ''))
                            
                            # Add phrase count info
                            if len(extracted_phrases) > 1:
                                reasoning += f" [Contains {len(extracted_phrases)} phrases total]"
                            
                            record.update({
                                
                                'phrase_index': phrase_idx,
                                'case_law_excerpt': case_law_excerpt,
                                'legislation_excerpt': legislation_excerpt,
                                'confidence': confidence,
                                'reasoning': reasoning,
                                'status': 'SUCCESS'
                            })
                            records.append(record.copy())
                        else:
                            # Non-dict phrase
                            record.update({
                                'phrase_index': phrase_idx,
                                'case_law_excerpt': str(phrase) if phrase else '',
                                'reasoning': f'Non-dictionary phrase format. Contains {len(extracted_phrases)} phrases.',
                                'status': 'NON_DICT_PHRASE'
                            })
                else:
                    record.update({
                        'reasoning': 'Empty extracted_phrases array',
                        'status': 'NO_PHRASES'
                    })
                
                successful_lines += 1
                
            except Exception as e:
                record.update({
                    'reasoning': f'Error processing parsed JSON: {str(e)[:100]}',
                    'status': 'DATA_EXTRACTION_ERROR'
                })
                failed_lines += 1
                failed_request_ids.append(custom_id)
            
            records.append(record)

    # Verification
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total JSONL lines: {total_lines}")
    print(f"Total CSV records: {len(records)}")
    print(f"Successful extractions: {successful_lines}")
    print(f"Failed extractions: {failed_lines}")
    
    if len(records) == total_lines:
        print(f"✅ Perfect match: {len(records)} records for {total_lines} lines")
    else:
        print(f"❌ MISMATCH: Expected {total_lines}, got {len(records)}")

    # Save results
    if records:
        df = pd.DataFrame(records)
        
        # Verify unique custom_ids
        unique_ids = df['custom_id'].nunique()
        print(f"Unique custom_ids: {unique_ids}")
        
        # Status breakdown
        print(f"\nStatus breakdown:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Save CSV
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\nSaved to: {output_csv_path}")
        
        # Save failed IDs
        if failed_request_ids:
            failed_ids_file = output_csv_path.replace('.csv', '_failed_ids.txt')
            with open(failed_ids_file, 'w', encoding='utf-8') as f:
                f.write(f"# Failed Request IDs - {len(failed_request_ids)} total\n\n")
                for req_id in failed_request_ids:
                    f.write(f"{req_id}\n")
            print(f"Failed IDs saved to: {failed_ids_file}")
        
        return df
    
    return pd.DataFrame()


#This is the function that is going to parse the final jsol sent to the claude for the decision to pick 1-2 best legislation excerpts
def extract_data_from_claude_final_decision(jsonl_path,output_csv_path):
    """
    Converts a JSONL file with `high_confidence_pairs` data into a CSV file.

    Args:
        jsonl_path (str): Path to the input JSONL file.
        csv_path (str): Path to the output CSV file.
    """
    rows = []

    # Read the JSONL file line by line
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue  # skip empty lines

            obj = json.loads(line)
            custom_id = obj.get('custom_id', 'UNKNOWN') 
            if 'result' in obj and 'message' in obj['result']:
                content = obj['result']['message'].get('content', '')
                if content and len(content) > 0:
                    data = content[0].get('text', {})
                    try:
                        data = clean_json_content(data)

                        data = json.loads(data) if isinstance(data, str) else data
                    except Exception as e:
                        try:
                            print(f"Error parsing JSON content: {e}")
                            data = data.strip()

                            # Remove non-JSON prefix junk (e.g., starting backticks or BOM chars)
                            data = re.sub(r'^[^{\[\"]+', '', data)  # remove anything before first {, [ or "
                            data = data.replace("`", "")  # remove stray backticks
                            data = data.replace("'", '"') # convert single to double quotes
                            
                            # Remove trailing commas
                            data = re.sub(r",\s*([}\]])", r"\1", data)
                            data = demjson3.decode(data, strict=False)
                        except Exception as e2:
                             print(f"Failed to parse JSON content after cleanup: {e2}")
                             rows.append({
                                "custom_id":custom_id,
                                "para_id": 'UNKNOWN',
                                "section_id": 'UNKNOWN',
                                "model": 'UNKNOWN',
                                "case_law_excerpt": 'UNKNOWN',
                                "legislation_excerpt": 'UNKNOWN'
                             })
                             continue

            para_id = data.get("para_id")
            high_conf_pairs = data.get("high_confidence_pairs", [])
            print("=======================================")
            print(f"Processing : {custom_id}")
            print(f"Found {len(high_conf_pairs)} high confidence pairs")
            for pair in high_conf_pairs:
                pair = json.loads(pair) if isinstance(pair, str) else pair
                rows.append({
                    "custom_id":custom_id,
                    "para_id": para_id,
                    "section_id": pair.get("section_id", ''),
                    "model": pair.get("model", ''),
                    "case_law_excerpt": pair.get("case_law_excerpt", ''),
                    "legislation_excerpt": pair.get("legislation_excerpt", '')
                })

    # Write to CSV using pandas DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")




def extract_comprehensive_with_fixing(output_jsonl_path, output_csv_path, source_csv_path):
    """
    Comprehensive extraction function that ensures exactly the expected number of records.
    First tries normal extraction, then fixes any failed records, and ensures proper status tracking.
    The expected count is automatically determined from the JSONL file length.
    
    Args:
        output_jsonl_path: Path to the output JSONL file
        output_csv_path: Path to save the final CSV
        source_csv_path: Path to source CSV with para_text and section_text
        
    Returns:
        DataFrame with exactly the number of records as requests in the JSONL file
    """
    import tempfile
    import os
    
    # Determine expected count from JSONL file
    print(f"🔍 Counting requests in JSONL file: {output_jsonl_path}")
    try:
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
        expected_count = line_count
        print(f"📊 Found {expected_count} requests in JSONL file")
    except Exception as e:
        print(f"❌ Error counting requests: {e}")
        expected_count = 18004  # Fallback
        print(f"⚠️ Using fallback count: {expected_count}")
    
    print(f"🔍 Starting comprehensive extraction for {expected_count} expected records...")
    
    # Step 1: Try normal extraction first
    print("\n📋 Step 1: Attempting normal extraction...")
    temp_error_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    temp_error_file.close()
    
    try:
        # Use the existing function but capture error records
        normal_df = extract_openai_output_to_csv(output_jsonl_path, output_csv_path, source_csv_path, temp_error_file.name)
        
        # Count successful records
        successful_count = len(normal_df) if not normal_df.empty else 0
        print(f"✅ Normal extraction completed: {successful_count} records")
        
        # Check if we have the expected count
        if successful_count >= expected_count:
            print(f"🎉 Success! Got {successful_count} records (expected {expected_count})")
            
            # Ensure we have exactly expected_count records
            if successful_count > expected_count:
                # Take first expected_count records
                final_df = normal_df.head(expected_count)
                final_df.to_csv(output_csv_path, index=False)
                print(f"📊 Truncated to exactly {expected_count} records")
            else:
                final_df = normal_df
                
            # Add status summary
            status_summary = final_df['status'].value_counts()
            print(f"\n📈 Status Summary:")
            for status, count in status_summary.items():
                print(f"   {status}: {count}")
                
            return final_df
            
    except Exception as e:
        print(f"❌ Normal extraction failed: {e}")
        successful_count = 0
    
    # Step 2: Fix failed records if needed
    print(f"\n🔧 Step 2: Processing failed records...")
    
    # Check if error file has content
    if os.path.exists(temp_error_file.name) and os.path.getsize(temp_error_file.name) > 0:
        print(f"📁 Found {os.path.getsize(temp_error_file.name)} bytes in error file")
        
        # Create temporary fixed file
        temp_fixed_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        temp_fixed_file.close()
        
        try:
            # Fix the JSON errors
            from fix_and_convert_jsonl import fix_jsonl_file, extract_data_from_fixed_jsonl, load_source_data
            
            # Load source data
            source_lookups = load_source_data(source_csv_path)
            
            # Fix JSON errors
            fix_stats = fix_jsonl_file(temp_error_file.name, temp_fixed_file.name)
            print(f"🔧 Fixed {fix_stats['fixed_lines']} out of {fix_stats['total_lines']} error records")
            
            # Extract from fixed file
            if os.path.exists(temp_fixed_file.name):
                fixed_records = extract_data_from_fixed_jsonl(temp_fixed_file.name, source_lookups)
                
                if fixed_records:
                    fixed_df = pd.DataFrame(fixed_records)
                    print(f"✅ Fixed extraction: {len(fixed_records)} records")
                    
                    # Combine with successful records
                    if successful_count > 0:
                        combined_df = pd.concat([normal_df, fixed_df], ignore_index=True)
                    else:
                        combined_df = fixed_df
                        
                    # Ensure we have exactly expected_count records
                    if len(combined_df) >= expected_count:
                        final_df = combined_df.head(expected_count)
                        final_df.to_csv(output_csv_path, index=False)
                        print(f"🎉 Success! Combined extraction: {len(combined_df)} records, using first {expected_count}")
                        
                        # Add status summary
                        status_summary = final_df['status'].value_counts()
                        print(f"\n📈 Status Summary:")
                        for status, count in status_summary.items():
                            print(f"   {status}: {count}")
                            
                        return final_df
                    else:
                        print(f"❌ Still missing records: {len(combined_df)} < {expected_count}")
                        # Create placeholder records for missing ones
                        missing_count = expected_count - len(combined_df)
                        placeholder_records = []
                        
                        for i in range(missing_count):
                            placeholder_records.append({
                                'para_id': f'MISSING_{i+1}',
                                'section_id': f'MISSING_{i+1}',
                                'para_text': '',
                                'section_text': '',
                                'case_law_excerpt': '',
                                'legislation_excerpt': '',
                                'confidence': '',
                                'reasoning': 'Record not found in extraction',
                                'phrase_index': -1,
                                'status': 'FAIL'
                            })
                        
                        placeholder_df = pd.DataFrame(placeholder_records)
                        final_df = pd.concat([combined_df, placeholder_df], ignore_index=True)
                        final_df.to_csv(output_csv_path, index=False)
                        print(f"📊 Added {missing_count} placeholder records to reach {expected_count}")
                        
                        # Add status summary
                        status_summary = final_df['status'].value_counts()
                        print(f"\n📈 Status Summary:")
                        for status, count in status_summary.items():
                            print(f"   {status}: {count}")
                            
                        return final_df
                else:
                    print("❌ No records extracted from fixed file")
            else:
                print("❌ Fixed file not created")
                
        except Exception as e:
            print(f"❌ Error during fixing process: {e}")
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_fixed_file.name)
            except:
                pass
    else:
        print("📁 No error records found")
    
    # Step 3: Create placeholder records if we still don't have enough
    print(f"\n📝 Step 3: Creating placeholder records...")
    
    if successful_count < expected_count:
        missing_count = expected_count - successful_count
        placeholder_records = []
        
        for i in range(missing_count):
            placeholder_records.append({
                'para_id': f'MISSING_{i+1}',
                'section_id': f'MISSING_{i+1}',
                'para_text': '',
                'section_text': '',
                'case_law_excerpt': '',
                'legislation_excerpt': '',
                'confidence': '',
                'reasoning': 'Record not found in extraction',
                'phrase_index': -1,
                'status': 'FAIL'
            })
        
        placeholder_df = pd.DataFrame(placeholder_records)
        
        if successful_count > 0:
            final_df = pd.concat([normal_df, placeholder_df], ignore_index=True)
        else:
            final_df = placeholder_df
            
        final_df.to_csv(output_csv_path, index=False)
        print(f"📊 Created {missing_count} placeholder records to reach {expected_count}")
        
        # Add status summary
        status_summary = final_df['status'].value_counts()
        print(f"\n📈 Status Summary:")
        for status, count in status_summary.items():
            print(f"   {status}: {count}")
            
        return final_df
    
    # Clean up temporary files
    try:
        os.unlink(temp_error_file.name)
    except:
        pass
    
    print("❌ Failed to extract expected number of records")
    return pd.DataFrame()
def extract_universal_output_to_csv(output_jsonl_path, output_csv_path, source_csv_path):
    """
    Universal extraction function that handles multiple response formats:
    - Deepseek/Groq format: response.body.choices[0].message.content
    - OpenAI format: response.body.choices[0].message.content  
    - Direct content format: response.content or content
    
    Ensures exactly one record per JSONL line (one per custom_id).
    
    Args:
        output_jsonl_path: Path to the output JSONL file
        output_csv_path: Path to save the extracted data as CSV
        source_csv_path: Path to source CSV with para_text and section_text
        
    Returns:
        DataFrame with exactly the number of records as lines in the JSONL file
    """
    # Load source CSV for text lookup
    try:
        source_df = pd.read_csv(source_csv_path, dtype=str)
        source_df = source_df.fillna('')
        
        # Dynamically find para_text and section_text columns
        para_col = None
        section_col = None
        
        for col in source_df.columns:
            if 'para' in col.lower() and 'text' in col.lower():
                para_col = col
            elif 'section' in col.lower() and 'text' in col.lower():
                section_col = col
        
        if not para_col:
            for col in source_df.columns:
                if 'para' in col.lower() and ('content' in col.lower() or 'paragraph' in col.lower()):
                    para_col = col
                    break
        
        if not section_col:
            for col in source_df.columns:
                if 'section' in col.lower() and 'content' in col.lower():
                    section_col = col
                    break
        
        print(f"Using columns: para_text='{para_col}', section_text='{section_col}'")
        
        # Create lookup dictionaries
        para_dict = {}
        section_dict = {}
        
        if para_col and 'para_id' in source_df.columns:
            para_dict = dict(zip(source_df['para_id'], source_df[para_col]))
        
        if section_col and 'section_id' in source_df.columns:
            section_dict = dict(zip(source_df['section_id'], source_df[section_col]))
            
    except Exception as e:
        print(f"Warning: Failed to load source CSV: {e}")
        para_dict = {}
        section_dict = {}

    def clean_json_content(content):
        """Clean JSON content by removing markdown formatting"""
        if not content:
            return content
        
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        content = re.sub(r'^```\s*', '', content)
        
        # Remove extra whitespace
        content = content.strip()
        
        return content

    def extract_content_from_response(obj):
        """Extract content from various response formats"""
        content = ''
        
        # Try different extraction paths
        extraction_paths = [
            # Deepseek/Groq format
            lambda x: x.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', ''),
            # OpenAI batch format
            lambda x: x.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', ''),
            # Direct response format
            lambda x: x.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content', ''),
            # Simple content format
            lambda x: x.get('response', {}).get('content', ''),
            lambda x: x.get('content', ''),
            # Message content directly
            lambda x: x.get('message', {}).get('content', ''),
            # Choices directly at root
            lambda x: x.get('choices', [{}])[0].get('message', {}).get('content', ''),
        ]
        
        for path_func in extraction_paths:
            try:
                content = path_func(obj)
                if content:
                    break
            except (KeyError, IndexError, TypeError):
                continue
        
        return content

    records = []
    total_lines = 0
    successful_lines = 0
    failed_lines = 0
    failed_request_ids = []

    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Initialize default values for this line - EVERY line gets a record
            custom_id = f'UNKNOWN_{line_num}'
            para_id = ''
            section_id = ''
            para_text = ''
            section_text = ''
            
            # Default record that will be created no matter what
            default_record = {
                'custom_id': custom_id,
                'para_id': para_id,
                'section_id': section_id,
                'phrase_index': -1,
                'case_law_excerpt': '',
                'legislation_excerpt': '',
                'confidence': '',
                'reasoning': 'Processing failed',
                'para_text': para_text,
                'section_text': section_text,
                'line_number': line_num,
                'status': 'PROCESSING_FAILED'
            }
            
            if not line.strip():
                # Empty line
                default_record.update({
                    'custom_id': f'EMPTY_LINE_{line_num}',
                    'reasoning': 'Empty line',
                    'status': 'EMPTY_LINE'
                })
                records.append(default_record)
                failed_lines += 1
                failed_request_ids.append(f'EMPTY_LINE_{line_num}')
                continue

            try:
                # Try to parse the basic JSON structure
                try:
                    obj = json.loads(line)
                    custom_id = obj.get('custom_id', f'UNKNOWN_{line_num}')
                    default_record['custom_id'] = custom_id
                except json.JSONDecodeError as e:
                    default_record.update({
                        'reasoning': f'Line JSON parse error: {str(e)[:200]}',
                        'status': 'LINE_JSON_ERROR'
                    })
                    records.append(default_record)
                    failed_lines += 1
                    failed_request_ids.append(custom_id)
                    continue
                
                # Extract content using universal extraction
                content = extract_content_from_response(obj)

                if not content:
                    default_record.update({
                        'reasoning': 'No content found in response',
                        'status': 'NO_CONTENT'
                    })
                    records.append(default_record)
                    failed_lines += 1
                    failed_request_ids.append(custom_id)
                    continue

                # Try to extract JSON from content
                try:
                    json_content = clean_json_content(content)
                    parsed_json = json.loads(json_content)
                    if isinstance(parsed_json, str):
                        parsed_json = json.loads(parsed_json)
                except json.JSONDecodeError as e:
                    default_record.update({
                        'reasoning': f'Content JSON parse error: {str(e)[:200]}',
                        'status': 'CONTENT_JSON_ERROR'
                    })
                    records.append(default_record)
                    failed_lines += 1
                    failed_request_ids.append(custom_id)
                    continue

                # Extract data from parsed JSON
                para_id = parsed_json.get('para_id', '')
                section_id = parsed_json.get('section_id', '')
                para_text = para_dict.get(para_id, '')
                section_text = section_dict.get(section_id, '')
                extracted_phrases = parsed_json.get('extracted_phrases', [])

                # Update default record with extracted data
                default_record.update({
                    'para_id': para_id,
                    'section_id': section_id,
                    'para_text': para_text,
                    'section_text': section_text
                })

                # Create ONE record per custom_id
                if extracted_phrases:
                    # Take the first phrase for main data
                    first_phrase = extracted_phrases[0] if extracted_phrases else {}
                    
                    if isinstance(first_phrase, dict):
                        case_law_excerpt = first_phrase.get('case_law_excerpt', '')
                        legislation_excerpt = first_phrase.get('legislation_excerpt', '')
                        confidence = first_phrase.get('confidence', '')
                        reasoning = first_phrase.get('reasoning', '')
                        
                        # If there are multiple phrases, mention in reasoning
                        if len(extracted_phrases) > 1:
                            reasoning += f" [Total phrases: {len(extracted_phrases)}]"
                        
                        default_record.update({
                            'phrase_index': 0,
                            'case_law_excerpt': case_law_excerpt,
                            'legislation_excerpt': legislation_excerpt,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'status': 'SUCCESS'
                        })
                    else:
                        # Non-dict first phrase
                        default_record.update({
                            'phrase_index': 0,
                            'case_law_excerpt': str(first_phrase) if first_phrase else '',
                            'reasoning': f'Non-dict phrase format [Total phrases: {len(extracted_phrases)}]',
                            'status': 'NON_DICT_PHRASE'
                        })
                else:
                    default_record.update({
                        'reasoning': 'No extracted_phrases found',
                        'status': 'NO_PHRASES'
                    })
                
                records.append(default_record)
                successful_lines += 1

            except Exception as e:
                # Catch-all for any unexpected errors
                default_record.update({
                    'reasoning': f'Unexpected error: {str(e)[:200]}',
                    'status': 'UNEXPECTED_ERROR'
                })
                records.append(default_record)
                failed_lines += 1
                failed_request_ids.append(custom_id)

    # Verify record count matches line count
    print(f"\nRecord Count Verification:")
    print(f"Total JSONL lines: {total_lines}")
    print(f"Total CSV records: {len(records)}")
    if len(records) != total_lines:
        print(f"⚠️  WARNING: Record count mismatch! Expected {total_lines}, got {len(records)}")
    else:
        print(f"✅ Record count matches JSONL line count")

    # Save failed request IDs to file
    failed_ids_file = output_csv_path.replace('.csv', '_failed_ids.txt')
    if failed_request_ids:
        try:
            with open(failed_ids_file, 'w', encoding='utf-8') as f:
                f.write(f"# Failed Request IDs from {output_jsonl_path}\n")
                f.write(f"# Total failed: {len(failed_request_ids)}\n")
                f.write(f"# Generated on: {pd.Timestamp.now()}\n\n")
                for req_id in failed_request_ids:
                    f.write(f"{req_id}\n")
            print(f"📄 Failed request IDs saved to: {failed_ids_file}")
        except Exception as e:
            print(f"❌ Error saving failed IDs file: {e}")

    # Save and return
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(['line_number', 'phrase_index'])
        df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"\nUniversal Extraction Summary:")
        print(f"Total lines processed: {total_lines}")
        print(f"Successful extractions: {successful_lines}")
        print(f"Failed extractions: {failed_lines}")
        print(f"Total records: {len(records)}")
        print(f"Unique custom_ids: {df['custom_id'].nunique()}")
        
        # Status breakdown
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"{status}: {count}")
        
        print(f"Saved to: {output_csv_path}")
        
        # Report failed request IDs
        if failed_request_ids:
            print(f"\n❌ Failed Request IDs ({len(failed_request_ids)}):")
            print(f"📄 Full list saved to: {failed_ids_file}")
            # Show first 10 failed IDs
            for req_id in failed_request_ids[:10]:
                print(f"   - {req_id}")
            if len(failed_request_ids) > 10:
                print(f"   ... and {len(failed_request_ids) - 10} more")
        else:
            print(f"\n✅ No failed requests!")

        return df
    else:
        print("No records extracted.")
        return pd.DataFrame()

def extract_combined_output_to_csv(output_jsonl_path, output_csv_path, source_csv_path):
    """
    Extract data from Deepseek combined output JSONL files.
    Deepseek format has response.body.choices[0].message.content structure.
    
    Args:
        output_jsonl_path: Path to the Deepseek output JSONL file
        output_csv_path: Path to save the extracted data as CSV
        source_csv_path: Path to source CSV with para_text and section_text
        
    Returns:
        DataFrame with extracted records
    """
    # Load source CSV for text lookup
    try:
        source_df = pd.read_csv(source_csv_path, dtype=str)
        source_df = source_df.fillna('')
        
        # Dynamically find para_text and section_text columns
        para_col = None
        section_col = None
        
        for col in source_df.columns:
            if 'para' in col.lower() and 'text' in col.lower():
                para_col = col
            elif 'section' in col.lower() and 'text' in col.lower():
                section_col = col
        
        if not para_col:
            for col in source_df.columns:
                if 'para' in col.lower() and ('content' in col.lower() or 'paragraph' in col.lower()):
                    para_col = col
                    break
        
        if not section_col:
            for col in source_df.columns:
                if 'section' in col.lower() and 'content' in col.lower():
                    section_col = col
                    break
        
        print(f"Using columns: para_text='{para_col}', section_text='{section_col}'")
        
        # Create lookup dictionaries
        para_dict = {}
        section_dict = {}
        
        if para_col and 'para_id' in source_df.columns:
            para_dict = dict(zip(source_df['para_id'], source_df[para_col]))
        
        if section_col and 'section_id' in source_df.columns:
            section_dict = dict(zip(source_df['section_id'], source_df[section_col]))
            
    except Exception as e:
        print(f"Warning: Failed to load source CSV: {e}")
        para_dict = {}
        section_dict = {}

    records = []
    total_lines = 0
    successful_lines = 0
    failed_lines = 0
    failed_request_ids = []

    with open(output_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            if not line.strip():
                # Even for empty lines, create a record to maintain count
                records.append({
                    'custom_id': f'EMPTY_LINE_{line_num}',
                    'para_id': '',
                    'section_id': '',
                    'phrase_index': -1,
                    'case_law_excerpt': '',
                    'legislation_excerpt': '',
                    'confidence': '',
                    'reasoning': 'Empty line',
                    'para_text': '',
                    'section_text': '',
                    'line_number': line_num,
                    'status': 'EMPTY_LINE'
                })
                failed_lines += 1
                failed_request_ids.append(f'EMPTY_LINE_{line_num}')
                continue

            # Initialize default values for this line
            custom_id = f'UNKNOWN_{line_num}'
            para_id = ''
            section_id = ''
            para_text = ''
            section_text = ''
            line_processed = False

            try:
                obj = json.loads(line)
                custom_id = obj.get('custom_id', f'UNKNOWN_{line_num}')
                
                # Extract content from Deepseek response structure
                content = ''
                try:
                    choices = obj.get('response', {}).get('body', {}).get('choices', [])
                    if choices and len(choices) > 0:
                        content = choices[0].get('message', {}).get('content', '')
                except (KeyError, IndexError):
                    content = obj.get('response', {}).get('content', '')
                    if not content:
                        content = obj.get('content', '')

                if not content:
                    print(f"No content found for custom_id: {custom_id} (line {line_num})")
                    records.append({
                        'custom_id': custom_id,
                        'para_id': para_id,
                        'section_id': section_id,
                        'phrase_index': -1,
                        'case_law_excerpt': '',
                        'legislation_excerpt': '',
                        'confidence': '',
                        'reasoning': 'No content found in response',
                        'para_text': para_text,
                        'section_text': section_text,
                        'line_number': line_num,
                        'status': 'NO_CONTENT'
                    })
                    failed_lines += 1
                    failed_request_ids.append(custom_id)
                    line_processed = True
                    continue

                # Try to extract JSON from content (Deepseek often wraps in markdown)
                json_content = clean_json_content(content)
                
                # Try standard JSON parsing
                parsed_json = None
                try:
                    parsed_json = json.loads(json_content)
                    if isinstance(parsed_json, str):
                        parsed_json = json.loads(parsed_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse content for custom_id {custom_id} (line {line_num}): {e}")
                    records.append({
                        'custom_id': custom_id,
                        'para_id': para_id,
                        'section_id': section_id,
                        'phrase_index': -1,
                        'case_law_excerpt': '',
                        'legislation_excerpt': '',
                        'confidence': '',
                        'reasoning': f'JSON parse error: {str(e)[:200]}',
                        'para_text': para_text,
                        'section_text': section_text,
                        'line_number': line_num,
                        'status': 'JSON_PARSE_ERROR'
                    })
                    failed_lines += 1
                    failed_request_ids.append(custom_id)
                    line_processed = True
                    continue

                # Extract data from parsed JSON
                para_id = parsed_json.get('para_id', '')
                section_id = parsed_json.get('section_id', '')
                para_text = para_dict.get(para_id, '')
                section_text = section_dict.get(section_id, '')
                extracted_phrases = parsed_json.get('extracted_phrases', [])

                if extracted_phrases:
                    for phrase_idx, phrase in enumerate(extracted_phrases):
                        if isinstance(phrase, dict):
                            records.append({
                                'custom_id': custom_id,
                                'para_id': para_id,
                                'section_id': section_id,
                                'phrase_index': phrase_idx,
                                'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                                'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                                'confidence': phrase.get('confidence', ''),
                                'reasoning': phrase.get('reasoning', ''),
                                'para_text': para_text,
                                'section_text': section_text,
                                'line_number': line_num,
                                'status': 'SUCCESS'
                            })
                        else:
                            records.append({
                                'custom_id': custom_id,
                                'para_id': para_id,
                                'section_id': section_id,
                                'phrase_index': phrase_idx,
                                'case_law_excerpt': str(phrase) if phrase else '',
                                'legislation_excerpt': '',
                                'confidence': '',
                                'reasoning': 'Non-dict phrase format',
                                'para_text': para_text,
                                'section_text': section_text,
                                'line_number': line_num,
                                'status': 'NON_DICT_PHRASE'
                            })
                else:
                    records.append({
                        'custom_id': custom_id,
                        'para_id': para_id,
                        'section_id': section_id,
                        'phrase_index': -1,
                        'case_law_excerpt': '',
                        'legislation_excerpt': '',
                        'confidence': '',
                        'reasoning': 'No extracted_phrases found',
                        'para_text': para_text,
                        'section_text': section_text,
                        'line_number': line_num,
                        'status': 'NO_PHRASES'
                    })
                
                successful_lines += 1
                line_processed = True

            except Exception as e:
                print(f"Unexpected error at line {line_num}: {e}")
                # Ensure we always create a record for this line, even on unexpected errors
                if not line_processed:
                    # Try to get custom_id from the raw line if possible
                    try:
                        temp_obj = json.loads(line)
                        custom_id = temp_obj.get('custom_id', f'UNKNOWN_{line_num}')
                    except:
                        custom_id = f'UNKNOWN_{line_num}'
                    
                    records.append({
                        'custom_id': custom_id,
                        'para_id': '',
                        'section_id': '',
                        'phrase_index': -1,
                        'case_law_excerpt': '',
                        'legislation_excerpt': '',
                        'confidence': '',
                        'reasoning': f'Unexpected error: {str(e)[:200]}',
                        'para_text': '',
                        'section_text': '',
                        'line_number': line_num,
                        'status': 'UNEXPECTED_ERROR'
                    })
                    failed_lines += 1
                    failed_request_ids.append(custom_id)

    # Verify record count matches line count
    print(f"\nRecord Count Verification:")
    print(f"Total JSONL lines: {total_lines}")
    print(f"Total CSV records: {len(records)}")
    if len(records) != total_lines:
        print(f"⚠️  WARNING: Record count mismatch! Expected {total_lines}, got {len(records)}")
    else:
        print(f"✅ Record count matches JSONL line count")

    # Save failed request IDs to file
    failed_ids_file = output_csv_path.replace('.csv', '_failed_ids.txt')
    if failed_request_ids:
        try:
            with open(failed_ids_file, 'w', encoding='utf-8') as f:
                f.write(f"# Failed Request IDs from {output_jsonl_path}\n")
                f.write(f"# Total failed: {len(failed_request_ids)}\n")
                f.write(f"# Generated on: {pd.Timestamp.now()}\n\n")
                for req_id in failed_request_ids:
                    f.write(f"{req_id}\n")
            print(f"📄 Failed request IDs saved to: {failed_ids_file}")
        except Exception as e:
            print(f"❌ Error saving failed IDs file: {e}")

    # Save and return
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(['line_number', 'phrase_index'])  # Sort by line number first to maintain order
        df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"\nDeepseek Extraction Summary:")
        print(f"Total lines processed: {total_lines}")
        print(f"Successful extractions: {successful_lines}")
        print(f"Failed extractions: {failed_lines}")
        print(f"Total records: {len(records)}")
        
        # Status breakdown
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"{status}: {count}")
        
        print(f"Saved to: {output_csv_path}")
        
        # Report failed request IDs
        if failed_request_ids:
            print(f"\n❌ Failed Request IDs ({len(failed_request_ids)}):")
            print(f"📄 Full list saved to: {failed_ids_file}")
            # Show first 10 failed IDs
            for req_id in failed_request_ids[:10]:
                print(f"   - {req_id}")
            if len(failed_request_ids) > 10:
                print(f"   ... and {len(failed_request_ids) - 10} more")
        else:
            print(f"\n✅ No failed requests!")

        return df
    else:
        print("No records extracted.")
        if failed_request_ids:
            print(f"\n❌ Failed Request IDs ({len(failed_request_ids)}):")
            print(f"📄 Full list saved to: {failed_ids_file}")
            for req_id in failed_request_ids:
                print(f"   - {req_id}")
        return pd.DataFrame()

#This function converts jsonl to csv for the claude model input sent for the decision
def extract_from_input_jsonl_claude_final(input_path, output_csv):
    """
    Reads a JSONL file containing request JSON objects with a consistent structure
    and extracts custom_id, para_id, para_text, section_id, section_text,
    model, case_law_excerpt, legislation_excerpt into a CSV.
    """
    records = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)

            custom_id = data.get("custom_id", "")
            body = data.get("body", {})
            messages = body.get("messages", [])

            # We assume the "user" message contains parsed JSON in the "content" field
            # and is stored as a list of dicts
            for msg in messages:
                if msg.get("role") == "user":
                    content_json = msg.get("content")
                    if isinstance(content_json, dict):
                        para_id = content_json.get("para_id", "")
                        para_text = content_json.get("para_text", "")
                        high_conf_pairs = content_json.get("high_confidence_pairs", [])

                        for pair in high_conf_pairs:
                            records.append({
                                "custom_id": custom_id,
                                "para_id": para_id,
                                "para_text": para_text,
                                "section_id": pair.get("section_id", ""),
                                "section_text": pair.get("section_text", ""),
                                "model": pair.get("model", ""),
                                "case_law_excerpt": pair.get("case_law_excerpt", ""),
                                "legislation_excerpt": pair.get("legislation_excerpt", "")
                            })
                    break

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
if __name__ == "__main__":
    # combined_csv_path = 'data/newData/combined.csv'  # Path to the uploaded combined.csv
    # output_jsonl_path = 'data/newData/output.jsonl'  # Path to the uploaded output.jsonl
    # output_csv_path = 'data/newData/merged_output.csv'  # Path where the output will be saved
    # columns_to_keep = ['para_id', 'if_law_applied', 'application_of_law_phrases', 'reason']
    
    # # Run the merge function
    # merge_combined_csv_and_output_jsonl(combined_csv_path, output_jsonl_path, output_csv_path)

    # input_jsonl_path = 'data/final_test/final/reexperiment/singlepara-deepseek-input.jsonl'
    # output_jsonl_path = 'data/final_test/final/reexperiment/output_batches/deepseek_output.jsonl'
    # output_csv_path = 'data/final_test/final/reexperiment/output_deepseek.csv'
    # extract_data_from_deepseek_jsonl_after_verification(input_jsonl_path, output_jsonl_path, output_csv_path)
    
    
    # input_jsonl_path = 'data/final_test/final/reexperiment/singlepara-llama-input.jsonl'
    # output_jsonl_path = 'data/final_test/final/reexperiment/output_batches/llama_output.jsonl'
    # output_csv_path = 'data/final_test/final/reexperiment/output_llama.csv'
    # extract_llama_jsonl_to_csv(input_jsonl_path, output_jsonl_path, output_csv_path)

    input_jsonl_path = 'data/final_test/final/reexperiment/fewhot/input_batches/llama_combined_input.jsonl'

    output_jsonl_path = 'data/final_test/final/reexperiment/fewhot/output_batches/openai_combined_output.jsonl'
    output_csv_path = 'data/final_test/final/reexperiment/fewhot/openai_combined_output.csv'

    output_llama_jsonl_path = 'data/final_test/final/reexperiment/fewhot/output_batches/llama_combined_output.jsonl'

    output_llama_csv_path = 'data/final_test/final/reexperiment/fewhot/llama_combined_output.csv'

    source_csv_path = 'data/final_test/final/reexperiment/combined_sourcedf_final_rebuild.csv'  # Path to source CSV with para_id and section_id texts

    
    
    output_deepseek_jsonl_path = 'data/final_test/final/reexperiment/fewhot/output_batches/deepseek_combined_output.jsonl'
    output_deepseek_csv_path = 'data/final_test/final/reexperiment/fewhot/deepseek_combined_output.csv'


    error_file_openai = 'data/final_test/final/reexperiment/fewhot/openai_error_requests.jsonl'
    error_file_llama = 'data/final_test/final/reexperiment/fewhot/llama_error_requests.jsonl'

    
    # Test the new comprehensive extraction function

    comprehensive_output_csv = 'data/final_test/final/reexperiment/fewhot/comprehensive_llama_output.csv'
    
 
    # print(f"\n✅ Comprehensive extraction completed!")
    # print(f"📊 Final DataFrame shape: {result_df.shape}")
    # print(f"📋 Columns: {list(result_df.columns)}")
    
    # extract_openai_output_to_csv(output_llama_jsonl_path, output_llama_csv_path,source_csv_path,error_file_llama)
    # extract_deepseek_data_improved(output_deepseek_jsonl_path, output_deepseek_csv_path,source_csv_path)
    
    #Test the new Deepseek combined output extraction function

    # output_jsonl_path_redo = 'data/final_test/final/reexperiment/fewhot/11August/df_source_all_low_claude_output.jsonl'
    # output_csv_path_redo = 'data/final_test/final/reexperiment/fewhot/11August/df_source_all_low_for_claude_output.csv'
    # source_csv_path_redo = 'data/final_test/final/reexperiment/fewhot/11August/df_source_all_low_for_claude.csv'

    # result_df = extract_universal_output_to_csv2(
    #     output_jsonl_path_redo, 
    #     output_csv_path_redo, 
    #     source_csv_path_redo
    # )
    
    # print(f"\n✅ Deepseek extraction completed!")
    # print(f"📊 Final DataFrame shape: {result_df.shape}")
    # print(f"📋 Columns: {list(result_df.columns)}")


    input_jsonl_path = 'data/final_test/final/reexperiment/fewhot/11August/claude_decision_to_pick_the_pairs.jsonl'
    output_csv_path = 'data/final_test/final/reexperiment/fewhot/11August/claude_decision_to_pick_the_pairs.csv'

    extract_data_from_claude_final_decision(input_jsonl_path, output_csv_path)


