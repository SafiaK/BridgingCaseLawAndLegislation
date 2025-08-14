import sys
import os
sys.path.append('src')

import pandas as pd
import json
from make_batch_jsonl_law_application import create_batch_jsonl_for_phrase_extraction

csv_path = 'data/final_test/final/reexperiment/fewhot/11August/df_source_all_low_for_claude.csv'
df = pd.read_csv(csv_path)

test_df = df.copy()
print(f"Testing with {len(test_df)} rows")
print(f"Sample para_ids: {list(test_df['para_id'])}")
print(f"Sample section_ids: {list(test_df['section_id'])}")

# Check data availability
print(f"\nData availability:")
print(f"Rows with section_text: {test_df['section_text'].notna().sum()}")
print(f"Rows with section_id: {test_df['section_id'].notna().sum()}")

# Set up paths
prompt_file = 'helper_data_files/redo_extraction_prompt.txt'  # Use the main prompt file
examples_file = 'helper_data_files/phrase_extraction_examples.json' 
# Set output path
output_path = 'data/final_test/final/reexperiment/fewhot/11August/df_source_all_low_for_claude.jsonl'

# Use your fine-tuned model
#model_name = "gpt-4o-mini"
#model_name = "llama-3.3-70b-versatile"
#model_name = "deepseek-r1-distill-llama-70b"
model_name = "claude-sonnet-4-20250514"  

print(f"\nConfiguration:")
print(f"Model: {model_name}")
print(f"Prompt file: {prompt_file}")
print(f"Examples file: {examples_file}")
print(f"Output path: {output_path}")

# Create the batch JSONL with correct prompt
try:
    created_files = create_batch_jsonl_for_phrase_extraction(
        model_name=model_name,
        prompt_file=prompt_file,
        examples_file=examples_file,
        df=test_df,
        output_path=output_path
    )
    
    print("\n✅ Successfully created JSONL files:")
    for file_path in created_files:
        print(f"   - {file_path}")
        
        # Show sample content to verify section_text is included
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            sample_data = json.loads(first_line)
            user_content = sample_data['body']['messages'][-1]['content']
            
            print(f"\n📋 Sample user content:")
            print(f"   {user_content[:300]}...")
            
            # Check if section_text and section_id are included
            if 'section_text:' in user_content:
                print(f"   ✅ section_text is included in the prompt")
            else:
                print(f"   ❌ section_text is missing from the prompt")
                
            if 'section_id:' in user_content:
                print(f"   ✅ section_id is included in the prompt")
            else:
                print(f"   ❌ section_id is missing from the prompt")
        
except Exception as e:
    print(f"❌ Error creating JSONL files: {e}")
    import traceback
    traceback.print_exc() 