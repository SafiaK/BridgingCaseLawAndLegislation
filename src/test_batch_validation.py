#!/usr/bin/env python3
"""
Test script for batch validation functionality.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from phrase_validator import PhraseValidator
import json

def analyze_paragraph_success_rate(jsonl_file):
    """
    Analyze how many para_ids got at least 1 valid extraction from the JSONL file.
    
    Args:
        jsonl_file (str): Path to the JSONL results file
        
    Returns:
        dict: Analysis results
    """
    # Parse the JSONL results directly
    results = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                results.append(data)
            except Exception as e:
                print(f"Error parsing JSON line: {e}")
                continue
    
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

def main():
    """Test the batch validation functionality."""
    
    # File paths
    jsonl_file = "data/final_test/final/combined_all_results.jsonl"
    csv_file = "data/final_test/final/withsectionpositvefinal_cleaned.csv"
    
    print("🔍 Testing Batch Validation Function")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL file not found: {jsonl_file}")
        return
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print(f"✅ Found JSONL file: {jsonl_file}")
    print(f"✅ Found CSV file: {csv_file}")
    
    # Run validation with direct format
    try:
        # Load the JSONL file directly
        results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                results.append(data)
        
        print(f"✅ Loaded {len(results)} results from JSONL")
        
        # Load original CSV data for validation
        print(f"📁 Loading original CSV data for validation...")
        original_df = pd.read_csv(csv_file)
        print(f"   Loaded {len(original_df)} rows from original CSV")
        
        # Create validation DataFrame with proper validation
        validation_data = []
        for result in results:
            para_id = result.get('para_id', '')
            section_id = result.get('section_id', '')
            extracted_phrases = result.get('extracted_phrases', [])
            
            # Find matching original data
            original_row = original_df[
                (original_df['para_id'] == para_id) & 
                (original_df['section_id'] == section_id)
            ]
            
            if original_row.empty:
                print(f"⚠️ No original data found for {para_id} | {section_id}")
                continue
            
            paragraph_text = original_row.iloc[0]['paragraphs']
            section_text = original_row.iloc[0]['section_text']
            
            for phrase in extracted_phrases:
                case_law_excerpt = phrase.get('case_law_excerpt', '')
                legislation_excerpt = phrase.get('legislation_excerpt', '')
                
                # Validate using PhraseValidator
                is_valid, reason, actual_case_law_sentence, actual_legislation_sentence = PhraseValidator.validate_phrase_match(
                    case_law_excerpt, 
                    legislation_excerpt, 
                    paragraph_text, 
                    section_text,
                    similarity_threshold=0.8
                )
                
                validation_data.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'case_law_excerpt': case_law_excerpt,
                    'legislation_excerpt': legislation_excerpt,
                    'confidence': phrase.get('confidence', ''),
                    'reasoning': phrase.get('reasoning', ''),
                    'is_valid': is_valid,
                    'reason': reason,
                    'actual_case_law_sentence': actual_case_law_sentence,
                    'actual_legislation_sentence': actual_legislation_sentence
                })
        
        validation_results = pd.DataFrame(validation_data)
        
        print("\n📊 Validation Results:")
        print("=" * 50)
        
        # Calculate summary manually
        total_phrases = len(validation_results)
        valid_phrases = len(validation_results[validation_results['is_valid'] == True])
        unique_para_ids = validation_results['para_id'].nunique()
        para_ids_with_valid = validation_results[validation_results['is_valid'] == True]['para_id'].nunique()
        
        summary = {
            'total_paragraphs': unique_para_ids,
            'total_phrases': total_phrases,
            'valid_phrases': valid_phrases,
            'invalid_phrases': total_phrases - valid_phrases,
            'success_rate': (valid_phrases / total_phrases * 100) if total_phrases > 0 else 0,
            'paragraphs_with_valid_phrases': para_ids_with_valid,
            'paragraph_success_rate': (para_ids_with_valid / unique_para_ids * 100) if unique_para_ids > 0 else 0
        }
        
        print(f"Total paragraphs processed: {summary['total_paragraphs']}")
        print(f"Total phrases extracted: {summary['total_phrases']}")
        print(f"Valid phrases: {summary['valid_phrases']}")
        print(f"Invalid phrases: {summary['invalid_phrases']}")
        print(f"Phrase success rate: {summary['success_rate']:.1f}%")
        print(f"Paragraphs with valid phrases: {summary['paragraphs_with_valid_phrases']}")
        print(f"Paragraph success rate: {summary['paragraph_success_rate']:.1f}%")
        
        # Show sample results
        print("\n📋 Sample Validation Results:")
        print("=" * 50)
        
        # Show valid extractions
        valid_results = validation_results[validation_results['is_valid'] == True]
        if not valid_results.empty:
            print(f"\n✅ Valid Extractions ({len(valid_results)}):")
            for idx, row in valid_results.head(3).iterrows():
                print(f"  Para ID: {row['para_id']}")
                print(f"  Section ID: {row['section_id']}")
                print(f"  Case Law: '{row['case_law_excerpt'][:100]}...'")
                print(f"  Legislation: '{row['legislation_excerpt'][:100]}...'")
                print(f"  Confidence: {row['confidence']}")
                print()
        
        # Show invalid extractions
        invalid_results = validation_results[validation_results['is_valid'] == False]
        if not invalid_results.empty:
            print(f"\n❌ Invalid Extractions ({len(invalid_results)}):")
            for idx, row in invalid_results.head(3).iterrows():
                print(f"  Para ID: {row['para_id']}")
                print(f"  Section ID: {row['section_id']}")
                print(f"  Reason: {row['reason']}")
                print()
        
        # Save results
        output_file = "data/final_test/validation_results.csv"
        validation_results.to_csv(output_file, index=False)
        print(f"\n💾 Saved validation results to: {output_file}")
        
        # Analyze paragraph success rate
        print("\n📊 Analyzing paragraph success rate...")
        paragraph_analysis = analyze_paragraph_success_rate(jsonl_file)
        
        print(f"\n📈 Paragraph Success Analysis:")
        print(f"  Total unique para_ids: {paragraph_analysis['total_para_ids']}")
        print(f"  Para_ids with at least 1 valid phrase: {paragraph_analysis['para_ids_with_valid_phrases']}")
        print(f"  Para_ids with no valid phrases: {paragraph_analysis['para_ids_without_valid_phrases']}")
        print(f"  Success rate: {paragraph_analysis['success_rate']:.1f}%")
        
        if paragraph_analysis['para_ids_without_valid_phrases'] > 0:
            print(f"\n❌ Para_ids with no valid phrases:")
            for para_id in paragraph_analysis['unique_para_ids']:
                if para_id not in paragraph_analysis['para_ids_with_valid']:
                    print(f"    - {para_id}")
        
        # Save summary
        summary_file = "data/final_test/validation_summary.csv"
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(summary_file, index=False)
        print(f"💾 Saved validation summary to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 