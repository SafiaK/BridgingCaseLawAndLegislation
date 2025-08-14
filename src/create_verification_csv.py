import pandas as pd
import json
import ast
import os
from pathlib import Path

def unpack_extracted_phrases_and_merge(df_path, original_data_path, output_path):
    """
    Unpack extracted_phrases from a dataframe, merge with original data, and create a verification-ready CSV.
    
    Args:
        df_path (str): Path to the input dataframe CSV from data_for_verification
        original_data_path (str): Path to withsectionpositvefinal_cleaned.csv
        output_path (str): Path for the output verification CSV
    """
    
    # Read the dataframe with extracted phrases
    print(f"Reading dataframe from: {df_path}")
    df = pd.read_csv(df_path)
    
    # Read the original data
    print(f"Reading original data from: {original_data_path}")
    original_df = pd.read_csv(original_data_path)
    
    # Initialize list to store unpacked data
    unpacked_data = []
    
    # Process each row
    for idx, row in df.iterrows():
        para_id = row['para_id']
        section_id = row['section_id']
        legislation = row['legislation']
        section = row['section']
        
        # Parse extracted_phrases
        extracted_phrases_str = row['extracted_phrases']
        
        # Handle empty extracted_phrases
        if pd.isna(extracted_phrases_str) or extracted_phrases_str == '[]':
            continue
            
        try:
            # Parse the string representation of the list of dictionaries
            extracted_phrases = ast.literal_eval(extracted_phrases_str)
            
            # Process each phrase
            for phrase in extracted_phrases:
                unpacked_row = {
                    'para_id': para_id,
                    'section_id': section_id,
                    'legislation': legislation,
                    'section': section,
                    'case_term': phrase.get('case_law_excerpt', ''),
                    'legislation_term': phrase.get('legislation_excerpt', ''),
                    'confidence': phrase.get('confidence', ''),
                    'reasoning': phrase.get('reasoning', ''),
                    'extracted_phrases_original': extracted_phrases_str  # Keep original for reference
                }
                unpacked_data.append(unpacked_row)
                
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing extracted_phrases for row {idx}: {e}")
            print(f"Problematic string: {extracted_phrases_str}")
            continue
    
    # Create unpacked dataframe
    unpacked_df = pd.DataFrame(unpacked_data)
    
    if len(unpacked_df) == 0:
        print("No extracted phrases found to process")
        return None
    
    print(f"Unpacked {len(unpacked_df)} phrases")
    
    # Debug: Check unpacked data structure
    print("DEBUG: Checking unpacked data structure...")
    print(f"Unpacked_df columns: {unpacked_df.columns.tolist()}")
    print(f"First few rows of unpacked_df:")
    print(unpacked_df.head())
    
    # Merge with original data on para_id
    print("Merging with original data...")
    print(f"Original_df columns: {original_df.columns.tolist()}")
    merged_df = pd.merge(unpacked_df, original_df, on='para_id', how='left')
    print(f"Merged_df columns: {merged_df.columns.tolist()}")
    print(f"First few rows of merged_df:")
    print(merged_df.head())
    print(f"Sample section_id_x values:")
    print(merged_df['section_id_x'].head())
    
    # Create the final verification dataframe with required columns
    verification_df = pd.DataFrame({
        'url': merged_df.get('case_uri', ''),  # Use case_uri as url
        'para_id': merged_df['para_id'],
        'paragraphs': merged_df.get('paragraphs', ''),
        'case_term_phrases': merged_df['extracted_phrases_original'],
        'legislation_id': merged_df['legislation'].str.replace('ukpga/', ''),
        'section_text': merged_df.get('section_text', ''),
        'case_term': merged_df['case_term'],
        'legislation_term': merged_df['legislation_term'],
        'confidence': merged_df['confidence_x'],  # From unpacked phrase (with _x suffix)
        'reasoning': merged_df['reasoning'],      # From unpacked phrase
        'key_phrases': merged_df['legislation_term'],  # Use legislation_excerpt as key_phrases
        'standardized_act_id': merged_df['section_id_x'].str.replace('id/ukpga/', '').str.replace('-', '_')  # Use section_id with prefix removed and hyphens replaced with underscores
    })
    
    # Save to CSV
    verification_df.to_csv(output_path, index=False)
    print(f"Created verification CSV with {len(verification_df)} rows at: {output_path}")
    
    return verification_df

def process_single_dataframe(df_filename, input_dir, original_data_path, output_dir):
    """
    Process a single dataframe from the data_for_verification directory.
    
    Args:
        df_filename (str): Name of the CSV file to process (e.g., 'ukpga_1963_2_analysis.csv')
        input_dir (str): Directory containing the analysis CSV files
        original_data_path (str): Path to withsectionpositvefinal_cleaned.csv
        output_dir (str): Directory to save the verification CSV files
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    input_path = os.path.join(input_dir, df_filename)
    output_filename = df_filename.replace('_analysis.csv', '_verification.csv')
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Processing: {df_filename}")
    try:
        verification_df = unpack_extracted_phrases_and_merge(input_path, original_data_path, output_path)
        if verification_df is not None:
            print(f"Successfully processed {len(verification_df)} verification rows")
        return verification_df
    except Exception as e:
        print(f"Error processing {df_filename}: {e}")
        return None

def list_available_dataframes(input_dir):
    """
    List all available dataframes in the input directory.
    
    Args:
        input_dir (str): Directory containing the analysis CSV files
    """
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_analysis.csv')]
    print(f"Available dataframes ({len(csv_files)} total):")
    for i, csv_file in enumerate(csv_files[:10], 1):  # Show first 10
        print(f"{i}. {csv_file}")
    if len(csv_files) > 10:
        print(f"... and {len(csv_files) - 10} more")
    return csv_files

if __name__ == "__main__":
    # Configuration
    input_directory = "data/final_test/final/data_for_verification"
    original_data_path = "data/final_test/final/withsectionpositvefinal_cleaned.csv"
    output_directory = "data/final_test/final/verification_csvs"
    
    # List available dataframes
    available_files = list_available_dataframes(input_directory)
    
    # Example: Process the first dataframe
    if available_files:
        first_df = available_files[0]
        print(f"\nProcessing first dataframe: {first_df}")
        result = process_single_dataframe(first_df, input_directory, original_data_path, output_directory)
        
        if result is not None:
            print(f"\nSample of verification data:")
            print(result.head())
    
    # To process a specific dataframe, uncomment and modify:
    specific_df = "ukpga_1989_41_analysis.csv"  # Replace with your desired filename
    process_single_dataframe(specific_df, input_directory, original_data_path, output_directory) 