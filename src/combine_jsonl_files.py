import json
import os
from typing import List

def combine_jsonl_files(input_files: List[str], output_file: str):
    """
    Combine multiple JSONL files into a single JSONL file.
    
    Args:
        input_files: List of paths to JSONL files to combine
        output_file: Path to the output combined JSONL file
    """
    print(f"🔗 Combining {len(input_files)} JSONL files into: {output_file}")
    
    combined_records = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: File not found: {file_path}")
            continue
            
        print(f"📖 Reading: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse each line as JSON
                        record = json.loads(line.strip())
                        combined_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Invalid JSON in {file_path} at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue
    
    print(f"📊 Total records to combine: {len(combined_records)}")
    
    # Write combined records to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in combined_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Successfully combined {len(combined_records)} records into: {output_file}")
        
    except Exception as e:
        print(f"❌ Error writing to {output_file}: {e}")
        raise

def combine_jsonl_files_binary(input_files: List[str], output_file: str):
    """
    Combine multiple JSONL files into a single JSONL file using binary mode.
    Useful for files that might have encoding issues.
    
    Args:
        input_files: List of paths to JSONL files to combine
        output_file: Path to the output combined JSONL file
    """
    print(f"🔗 Combining {len(input_files)} JSONL files (binary mode) into: {output_file}")
    
    combined_records = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: File not found: {file_path}")
            continue
            
        print(f"📖 Reading: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Decode and parse each line as JSON
                        line_str = line.decode('utf-8').strip()
                        if line_str:  # Skip empty lines
                            record = json.loads(line_str)
                            combined_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Invalid JSON in {file_path} at line {line_num}: {e}")
                        continue
                    except UnicodeDecodeError as e:
                        print(f"⚠️  Warning: Encoding issue in {file_path} at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue
    
    print(f"📊 Total records to combine: {len(combined_records)}")
    
    # Write combined records to output file
    try:
        with open(output_file, 'wb') as f:
            for record in combined_records:
                line = json.dumps(record, ensure_ascii=False) + '\n'
                f.write(line.encode('utf-8'))
        
        print(f"✅ Successfully combined {len(combined_records)} records into: {output_file}")
        
    except Exception as e:
        print(f"❌ Error writing to {output_file}: {e}")
        raise

def combine_jsonl_files_with_validation(input_files: List[str], output_file: str, validate_func=None):
    """
    Combine multiple JSONL files with optional validation.
    
    Args:
        input_files: List of paths to JSONL files to combine
        output_file: Path to the output combined JSONL file
        validate_func: Optional function to validate each record before including
    """
    print(f"🔗 Combining {len(input_files)} JSONL files with validation into: {output_file}")
    
    combined_records = []
    skipped_records = 0
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: File not found: {file_path}")
            continue
            
        print(f"📖 Reading: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        
                        # Apply validation if provided
                        if validate_func:
                            if validate_func(record):
                                combined_records.append(record)
                            else:
                                skipped_records += 1
                        else:
                            combined_records.append(record)
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Invalid JSON in {file_path} at line {line_num}: {e}")
                        skipped_records += 1
                        continue
                        
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue
    
    print(f"📊 Total records combined: {len(combined_records)}")
    print(f"📊 Skipped records: {skipped_records}")
    
    # Write combined records to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in combined_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Successfully combined {len(combined_records)} records into: {output_file}")
        
    except Exception as e:
        print(f"❌ Error writing to {output_file}: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Example: Combine multiple JSONL files
    input_files = [
        "data/final_test/final/reexperiment/fewhot/input_batches/deepseek-input_part_01.jsonl",
        "data/final_test/final/reexperiment/fewhot/input_batches/deepseek-input_part_02.jsonl"
    ]
    
    output_file = "data/final_test/final/reexperiment/fewhot/input_batches/combined_deepseek.jsonl"
    
    # Combine files
    combine_jsonl_files(input_files, output_file)
    
    print(f"\n📋 Summary:")
    print(f"   Input files: {len(input_files)}")
    print(f"   Output file: {output_file}") 