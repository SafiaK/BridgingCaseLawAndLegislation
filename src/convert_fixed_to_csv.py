import json
import pandas as pd
import os
from typing import List, Dict, Any

def extract_data_from_fixed_jsonl(jsonl_file: str) -> List[Dict[str, Any]]:
    """
    Extract data from fixed JSONL files and convert to list of dictionaries.
    
    Args:
        jsonl_file: Path to the fixed JSONL file
        
    Returns:
        List of dictionaries with extracted data
    """
    records = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Extract para_id and section_id
                para_id = data.get('custom_id', '')
                
                # Get content from response
                content = ''
                try:
                    choices = data.get('response', {}).get('body', {}).get('choices', [])
                    if choices and len(choices) > 0:
                        content = choices[0].get('message', {}).get('content', '')
                except (KeyError, IndexError):
                    content = data.get('response', {}).get('content', '')
                    if not content:
                        content = data.get('content', '')
                
                if not content:
                    continue
                
                # Parse the content JSON
                try:
                    content_data = json.loads(content)
                    section_id = content_data.get('section_id', '')
                    extracted_phrases = content_data.get('extracted_phrases', [])
                    
                    # Process each phrase
                    for phrase_idx, phrase in enumerate(extracted_phrases):
                        if isinstance(phrase, dict):
                            record = {
                                'para_id': para_id,
                                'section_id': section_id,
                                'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                                'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                                'confidence': phrase.get('confidence', ''),
                                'reasoning': phrase.get('reasoning', ''),
                                'phrase_index': phrase_idx
                            }
                            records.append(record)
                        else:
                            # Handle non-dict phrases
                            record = {
                                'para_id': para_id,
                                'section_id': section_id,
                                'case_law_excerpt': str(phrase) if phrase else '',
                                'legislation_excerpt': '',
                                'confidence': '',
                                'reasoning': 'Non-dict phrase format',
                                'phrase_index': phrase_idx
                            }
                            records.append(record)
                    
                    # If no phrases, create a record with empty phrases
                    if not extracted_phrases:
                        record = {
                            'para_id': para_id,
                            'section_id': section_id,
                            'case_law_excerpt': '',
                            'legislation_excerpt': '',
                            'confidence': '',
                            'reasoning': '',
                            'phrase_index': 0
                        }
                        records.append(record)
                        
                except json.JSONDecodeError:
                    # If content is not valid JSON, try to extract using regex
                    import re
                    
                    para_id_match = re.search(r'"para_id"\s*:\s*"([^"]+)"', content)
                    section_id_match = re.search(r'"section_id"\s*:\s*"([^"]+)"', content)
                    
                    if para_id_match and section_id_match:
                        # Extract phrases using regex
                        phrase_blocks = re.findall(r'\{[^{}]*"case_law_excerpt"[^{}]*\}', content)
                        
                        if phrase_blocks:
                            for phrase_idx, block in enumerate(phrase_blocks):
                                case_law = re.search(r'"case_law_excerpt"\s*:\s*"([^"]*)"', block)
                                legislation = re.search(r'"legislation_excerpt"\s*:\s*"([^"]*)"', block)
                                reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)"', block)
                                confidence = re.search(r'"confidence"\s*:\s*"([^"]*)"', block)
                                
                                record = {
                                    'para_id': para_id_match.group(1),
                                    'section_id': section_id_match.group(1),
                                    'case_law_excerpt': case_law.group(1) if case_law else '',
                                    'legislation_excerpt': legislation.group(1) if legislation else '',
                                    'confidence': confidence.group(1) if confidence else '',
                                    'reasoning': reasoning.group(1) if reasoning else '',
                                    'phrase_index': phrase_idx
                                }
                                records.append(record)
                        else:
                            # No phrases found
                            record = {
                                'para_id': para_id_match.group(1),
                                'section_id': section_id_match.group(1),
                                'case_law_excerpt': '',
                                'legislation_excerpt': '',
                                'confidence': '',
                                'reasoning': '',
                                'phrase_index': 0
                            }
                            records.append(record)
                            
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return records

def convert_fixed_files_to_csv():
    """Convert fixed JSONL files to CSV for verification."""
    
    # Define file paths
    llama_fixed_file = 'data/final_test/final/reexperiment/fewhot/llama_error_requests_fixed.jsonl'
    openai_fixed_file = 'data/final_test/final/reexperiment/fewhot/openai_error_requests_fixed.jsonl'
    
    llama_csv_file = 'data/final_test/final/reexperiment/fewhot/llama_fixed_verification.csv'
    openai_csv_file = 'data/final_test/final/reexperiment/fewhot/openai_fixed_verification.csv'
    
    print("🔄 Converting fixed JSONL files to CSV for verification...")
    
    # Process Llama fixed file
    if os.path.exists(llama_fixed_file):
        print(f"\n📁 Processing Llama fixed file: {llama_fixed_file}")
        llama_records = extract_data_from_fixed_jsonl(llama_fixed_file)
        
        if llama_records:
            llama_df = pd.DataFrame(llama_records)
            llama_df.to_csv(llama_csv_file, index=False)
            print(f"✅ Llama CSV created:")
            print(f"   Records: {len(llama_records)}")
            print(f"   Saved to: {llama_csv_file}")
            print(f"   Columns: {list(llama_df.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample Llama data:")
            print(llama_df.head(3).to_string())
        else:
            print("❌ No valid records found in Llama fixed file")
    else:
        print(f"❌ Llama fixed file not found: {llama_fixed_file}")
    
    # Process OpenAI fixed file
    if os.path.exists(openai_fixed_file):
        print(f"\n📁 Processing OpenAI fixed file: {openai_fixed_file}")
        openai_records = extract_data_from_fixed_jsonl(openai_fixed_file)
        
        if openai_records:
            openai_df = pd.DataFrame(openai_records)
            openai_df.to_csv(openai_csv_file, index=False)
            print(f"✅ OpenAI CSV created:")
            print(f"   Records: {len(openai_records)}")
            print(f"   Saved to: {openai_csv_file}")
            print(f"   Columns: {list(openai_df.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample OpenAI data:")
            print(openai_df.head(3).to_string())
        else:
            print("❌ No valid records found in OpenAI fixed file")
    else:
        print(f"❌ OpenAI fixed file not found: {openai_fixed_file}")
    
    print("\n🎉 CSV conversion completed!")

if __name__ == "__main__":
    convert_fixed_files_to_csv() 