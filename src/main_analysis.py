import sys
import os
sys.path.append('src')

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from pathlib import Path
# Import PhraseValidator
from src.phrase_validator import PhraseValidator
class DatasetAnalyzer:
    def __init__(self):
        self.combined_results = []
        self.original_data = None
        self.validation_results = None
        
    def load_original_data(self):
        """Load the original CSV data"""
        csv_path = 'data/final_test/final/withsectionpositvefinal_cleaned.csv'
        self.original_data = pd.read_csv(csv_path)
        print(f"📊 Loaded original data: {len(self.original_data)} rows")
        return self.original_data
    
    def combine_results_with_custom_id_matching(self):
        """Combine results using custom_id matching to prevent hallucination issues"""
        
        # Define result and request file pairs
        file_pairs = [
            # Include batch_2_requests (first 100)
            ('data/final_test/final/batch_2_requests_results.jsonl', None),  # No request file needed for OpenAI format
            ('data/final_test/final/remaining_dataset_results_part_01.jsonl', 
             'data/final_test/final/remaining_dataset_batch_part_01.jsonl'),
            ('data/final_test/final/remaining_dataset_results_part_02.jsonl', 
             'data/final_test/final/remaining_dataset_batch_part_02.jsonl')
        ]
        
        all_results = []
        
        for result_file, request_file in file_pairs:
            if not os.path.exists(result_file):
                print(f"⚠️ Result file not found: {result_file}")
                continue
            
            print(f"📂 Processing: {result_file}")
            
            if request_file is None:
                # Handle OpenAI batch format (like batch_2_requests_results.jsonl)
                print(f"🔗 Processing OpenAI batch format (no request file needed)")
                
                matched_results = 0
                with open(result_file, 'r') as f:
                    for line in f:
                        result_data = json.loads(line)
                        custom_id = result_data.get('custom_id', '')
                        
                        try:
                            # Extract LLM response
                            response_content = result_data['response']['body']['choices'][0]['message']['content']
                            extracted_data = json.loads(response_content)
                            
                            # For OpenAI batch format, trust the LLM's para_id and section_id
                            # (since we don't have request file to verify against)
                            corrected_result = {
                                'para_id': extracted_data.get('para_id', ''),
                                'section_id': extracted_data.get('section_id', ''),
                                'extracted_phrases': extracted_data.get('extracted_phrases', []),
                                'reason': extracted_data.get('reason', ''),
                                'custom_id': custom_id,
                                'llm_para_id': extracted_data.get('para_id', ''),
                                'llm_section_id': extracted_data.get('section_id', ''),
                                'source': 'openai_batch'
                            }
                            
                            all_results.append(corrected_result)
                            matched_results += 1
                            
                        except Exception as e:
                            print(f"  ⚠️ Error parsing result for {custom_id}: {e}")
                            continue
                
                print(f"  ✅ Processed {matched_results} OpenAI batch results")
                
            else:
                # Handle custom format with request file matching
                if not os.path.exists(request_file):
                    print(f"⚠️ Request file not found: {request_file}")
                    continue
                
                print(f"🔗 Matching with: {request_file}")
                
                # Load request data to create custom_id mapping
                request_mapping = {}
                with open(request_file, 'r') as f:
                    for line in f:
                        req_data = json.loads(line)
                        custom_id = req_data.get('custom_id', '')
                        user_content = req_data['body']['messages'][1]['content']  # User message
                        
                        # Extract para_id and section_id from user content
                        lines = user_content.split('\n')
                        para_id = lines[0].replace('para_id: ', '') if len(lines) > 0 else ''
                        section_id = lines[3].replace('section_id: ', '') if len(lines) > 3 else ''
                        
                        request_mapping[custom_id] = {
                            'para_id': para_id,
                            'section_id': section_id,
                            'user_content': user_content
                        }
                
                print(f"  📋 Loaded {len(request_mapping)} request mappings")
                
                # Load results and match with requests
                matched_results = 0
                with open(result_file, 'r') as f:
                    for line in f:
                        result_data = json.loads(line)
                        custom_id = result_data.get('custom_id', '')
                        
                        if custom_id not in request_mapping:
                            print(f"  ⚠️ No request mapping found for custom_id: {custom_id}")
                            continue
                        
                        try:
                            # Extract LLM response
                            response_content = result_data['response']['body']['choices'][0]['message']['content']
                            extracted_data = json.loads(response_content)
                            
                            # Use original para_id and section_id from request (not LLM output)
                            original_para_id = request_mapping[custom_id]['para_id']
                            original_section_id = request_mapping[custom_id]['section_id']
                            
                            # Create corrected result
                            corrected_result = {
                                'para_id': original_para_id,  # Use original, not LLM output
                                'section_id': original_section_id,  # Use original, not LLM output
                                'extracted_phrases': extracted_data.get('extracted_phrases', []),
                                'reason': extracted_data.get('reason', ''),
                                'custom_id': custom_id,
                                'llm_para_id': extracted_data.get('para_id', ''),  # Keep LLM's version for comparison
                                'llm_section_id': extracted_data.get('section_id', ''),  # Keep LLM's version for comparison
                                'source': 'custom_matched'
                            }
                            
                            all_results.append(corrected_result)
                            matched_results += 1
                            
                        except Exception as e:
                            print(f"  ⚠️ Error parsing result for {custom_id}: {e}")
                            continue
                
                print(f"  ✅ Matched {matched_results} results")
        
        self.combined_results = all_results
        print(f"✅ Combined {len(self.combined_results)} results using custom_id matching")
        return all_results
    
    def combine_all_results(self):
        """Combine results from all batch files using custom_id matching"""
        return self.combine_results_with_custom_id_matching()
    
    def save_combined_results(self):
        """Save combined results to a single JSONL file and delete old file"""
        output_path = 'data/final_test/final/combined_all_results.jsonl'
        
        # Delete old combined file to avoid duplicates
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"🗑️ Deleted old combined results file")
        
        with open(output_path, 'w') as f:
            for result in self.combined_results:
                f.write(json.dumps(result) + '\n')
        
        print(f"💾 Saved combined results to: {output_path}")
        return output_path
    
    def analyze_missing_results(self):
        """Analyze what happened to missing results"""
        print(f"\n📊 MISSING RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        if self.original_data is None:
            self.load_original_data()
        
        # Get all original para_ids
        all_original_para_ids = set(self.original_data['para_id'].unique())
        extracted_para_ids = set(r.get('para_id', '') for r in self.combined_results if r.get('para_id'))
        missing_para_ids = all_original_para_ids - extracted_para_ids
        
        print(f"Total original para_ids: {len(all_original_para_ids)}")
        print(f"Para_ids with extractions: {len(extracted_para_ids)}")
        print(f"Missing para_ids: {len(missing_para_ids)}")
        
        if missing_para_ids:
            print(f"\n❌ Missing para_ids (first 10):")
            for para_id in list(missing_para_ids)[:10]:
                print(f"  - {para_id}")
        
        return {
            'total_original': len(all_original_para_ids),
            'extracted': len(extracted_para_ids),
            'missing': len(missing_para_ids),
            'missing_para_ids': list(missing_para_ids)
        }
    
    def save_comprehensive_results(self):
        """Save all analysis results to final_test/final/results folder"""
        results_dir = Path('data/final_test/final/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 SAVING COMPREHENSIVE RESULTS")
        print(f"{'='*60}")
        print(f"📁 Results directory: {results_dir}")
        
        # 1. Save dataset coverage analysis
        coverage_analysis = self.analyze_dataset_coverage()
        with open(results_dir / 'dataset_coverage_analysis.json', 'w') as f:
            json.dump(coverage_analysis, f, indent=2)
        print(f"✅ Saved dataset coverage analysis")
        
        # 2. Save extraction quality analysis  
        quality_analysis = self.analyze_extraction_quality()
        with open(results_dir / 'extraction_quality_analysis.json', 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            quality_for_json = quality_analysis.copy()
            if 'validation_results' in quality_for_json and hasattr(quality_for_json['validation_results'], 'to_dict'):
                quality_for_json['validation_results'] = quality_for_json['validation_results'].to_dict('records')
            json.dump(quality_for_json, f, indent=2)
        print(f"✅ Saved extraction quality analysis")
        
        # 3. Save missing results analysis
        missing_analysis = self.analyze_missing_results()
        with open(results_dir / 'missing_results_analysis.json', 'w') as f:
            json.dump(missing_analysis, f, indent=2)
        print(f"✅ Saved missing results analysis")
        
        # 4. Save research insights
        research_insights = self.generate_research_insights()
        with open(results_dir / 'research_insights.json', 'w') as f:
            json.dump(research_insights, f, indent=2)
        print(f"✅ Saved research insights")
        
        # 5. Save failure examples
        self.save_failure_examples(results_dir, quality_analysis.get('validation_results'))
        
        # 6. Save valid/invalid examples
        self.save_validation_examples(results_dir, quality_analysis.get('validation_results'))
        
        print(f"\n✅ All results saved to: {results_dir}")
        return results_dir
    
    def save_failure_examples(self, results_dir, validation_results):
        """Save examples of different types of failures"""
        if validation_results is None or validation_results.empty:
            return
        
        failure_examples = {
            'case_law_not_found': [],
            'legislation_not_found': [],
            'both_not_found': [],
            'other_failures': []
        }
        
        invalid_results = validation_results[validation_results['is_valid'] == False]
        
        for _, row in invalid_results.iterrows():
            example = {
                'para_id': row['para_id'],
                'section_id': row['section_id'],
                'case_law_excerpt': row['case_law_excerpt'],
                'legislation_excerpt': row['legislation_excerpt'],
                'confidence': row['confidence'],
                'llm_reasoning': row.get('reasoning', ''),  # LLM's reasoning for the extraction
                'validation_reason': row['reason'],  # Why validation failed
                'custom_id': row.get('custom_id', ''),
                'actual_case_law_sentence': row.get('actual_case_law_sentence', ''),  # Matching case law sentence
                'actual_legislation_sentence': row.get('actual_legislation_sentence', '')  # Matching legislation sentence
            }
            
            reason = row['reason'].lower()
            if 'case law' in reason and 'not found' in reason:
                if len(failure_examples['case_law_not_found']) < 3:
                    failure_examples['case_law_not_found'].append(example)
            elif 'legislation' in reason and 'not found' in reason:
                if len(failure_examples['legislation_not_found']) < 3:
                    failure_examples['legislation_not_found'].append(example)
            elif 'both' in reason or ('case law' in reason and 'legislation' in reason):
                if len(failure_examples['both_not_found']) < 3:
                    failure_examples['both_not_found'].append(example)
            else:
                if len(failure_examples['other_failures']) < 3:
                    failure_examples['other_failures'].append(example)
        
        with open(results_dir / 'failure_examples.json', 'w') as f:
            json.dump(failure_examples, f, indent=2)
        print(f"✅ Saved failure examples")
    
    def save_validation_examples(self, results_dir, validation_results):
        """Save examples of valid and invalid extractions"""
        if validation_results is None or validation_results.empty:
            return
        
        examples = {
            'valid_examples': [],
            'invalid_examples': []
        }
        
        # Get valid examples
        valid_results = validation_results[validation_results['is_valid'] == True]
        for _, row in valid_results.head(5).iterrows():
            examples['valid_examples'].append({
                'para_id': row['para_id'],
                'section_id': row['section_id'],
                'case_law_excerpt': row['case_law_excerpt'],
                'legislation_excerpt': row['legislation_excerpt'],
                'confidence': row['confidence'],
                'reasoning': row.get('reasoning', ''),
                'custom_id': row.get('custom_id', '')
            })
        
        # Get invalid examples
        invalid_results = validation_results[validation_results['is_valid'] == False]
        for _, row in invalid_results.head(5).iterrows():
            examples['invalid_examples'].append({
                'para_id': row['para_id'],
                'section_id': row['section_id'],
                'case_law_excerpt': row['case_law_excerpt'],
                'legislation_excerpt': row['legislation_excerpt'],
                'confidence': row['confidence'],
                'reason': row['reason'],
                'custom_id': row.get('custom_id', '')
            })
        
        with open(results_dir / 'validation_examples.json', 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"✅ Saved validation examples")
    
    def analyze_dataset_coverage(self):
        """Analyze how many para_ids have sections and extractions"""
        if self.original_data is None:
            self.load_original_data()
        
        # Analysis 1: Section distribution
        section_counts = self.original_data['para_id'].value_counts()
        
        print(f"\n📊 DATASET COVERAGE ANALYSIS")
        print(f"{'='*60}")
        print(f"Total unique para_ids: {len(section_counts)}")
        print(f"Total rows in dataset: {len(self.original_data)}")
        
        # Section distribution
        section_distribution = section_counts.value_counts().sort_index()
        print(f"\n📋 Section Distribution per para_id:")
        for sections, count in section_distribution.items():
            print(f"  {sections} section(s): {count} para_ids")
        
        # Analysis 2: Extraction coverage
        extracted_para_ids = set()
        for result in self.combined_results:
            para_id = result.get('para_id', '')
            if para_id:
                extracted_para_ids.add(para_id)
        
        print(f"\n📋 Extraction Coverage:")
        print(f"  Para_ids with extractions: {len(extracted_para_ids)}")
        print(f"  Para_ids without extractions: {len(section_counts) - len(extracted_para_ids)}")
        print(f"  Extraction success rate: {len(extracted_para_ids)/len(section_counts)*100:.1f}%")
        
        # Find para_ids without extractions
        all_para_ids = set(section_counts.index)
        missing_para_ids = all_para_ids - extracted_para_ids
        
        if missing_para_ids:
            print(f"\n❌ Para_ids missing extractions (first 10):")
            for para_id in list(missing_para_ids)[:10]:
                print(f"  - {para_id}")
        
        return {
            'total_para_ids': len(section_counts),
            'total_rows': len(self.original_data),
            'section_distribution': section_distribution.to_dict(),
            'extracted_para_ids': len(extracted_para_ids),
            'missing_para_ids': len(missing_para_ids),
            'success_rate': len(extracted_para_ids)/len(section_counts)*100
        }
    
   
    
    def analyze_extraction_quality(self):
        """Analyze the quality of extractions using the combined file and test_batch_validation approach"""
        print(f"\n📊 EXTRACTION QUALITY ANALYSIS")
        print(f"{'='*60}")
        
        # Use the test_batch_validation approach on the combined file
        combined_jsonl = 'data/final_test/final/combined_all_results.jsonl'
        csv_file = 'data/final_test/final/withsectionpositvefinal_cleaned.csv'
        
        print(f"📁 Using combined JSONL: {combined_jsonl}")
        print(f"📁 Using CSV: {csv_file}")
        
        # Check if files exist
        if not os.path.exists(combined_jsonl):
            print(f"❌ Combined JSONL file not found: {combined_jsonl}")
            return {'error': 'Combined JSONL file not found'}
        
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            return {'error': 'CSV file not found'}
        
        # Load the JSONL file directly (same as test_batch_validation)
        results = []
        with open(combined_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                results.append(data)
        
        print(f"✅ Loaded {len(results)} results from combined JSONL")
        
        # Load original CSV data for validation
        print(f"📁 Loading original CSV data for validation...")
        original_df = pd.read_csv(csv_file)
        print(f"   Loaded {len(original_df)} rows from original CSV")
        
        # Create validation DataFrame with proper validation and track empty extractions
        validation_data = []
        empty_extractions_count = 0
        empty_extractions_examples = []
        
        for result in results:
            para_id = result.get('para_id', '')
            section_id = result.get('section_id', '')
            extracted_phrases = result.get('extracted_phrases', [])
            reason = result.get('reason', '')
            
            # Check for empty extracted_phrases
            if not extracted_phrases or len(extracted_phrases) == 0:
                empty_extractions_count += 1
                if len(empty_extractions_examples) < 5:  # Keep first 5 examples
                    empty_extractions_examples.append({
                        'para_id': para_id,
                        'section_id': section_id,
                        'reason': reason,
                        'custom_id': result.get('custom_id', '')
                    })
                continue  # Skip validation for empty extractions
            
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
                
                # Validate using PhraseValidator (with correct parameters)
                is_valid, reason, actual_case_law_sentence, actual_legislation_sentence = PhraseValidator.validate_phrase_match(
                    case_law_excerpt, 
                    legislation_excerpt, 
                    paragraph_text, 
                    section_text,
                    similarity_threshold=1.0
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
                    'custom_id': result.get('custom_id', ''),
                    'actual_case_law_sentence': actual_case_law_sentence,
                    'actual_legislation_sentence': actual_legislation_sentence
                })
        
        validation_results = pd.DataFrame(validation_data)
        
        if validation_results.empty:
            print("❌ No validation results generated")
            return {'error': 'No validation results'}
        
        # Calculate summary manually (same as test_batch_validation)
        total_phrases = len(validation_results)
        valid_phrases = len(validation_results[validation_results['is_valid'] == True])
        unique_para_ids = validation_results['para_id'].nunique()
        para_ids_with_valid = validation_results[validation_results['is_valid'] == True]['para_id'].nunique()
        
        print(f"\n📊 Validation Summary:")
        print(f"  Total results processed: {len(results)}")
        print(f"  Empty extractions (no phrases): {empty_extractions_count}")
        print(f"  Results with phrases: {len(results) - empty_extractions_count}")
        print(f"  Total phrases extracted: {total_phrases}")
        print(f"  Valid phrases: {valid_phrases}")
        print(f"  Invalid phrases: {total_phrases - valid_phrases}")
        print(f"  Phrase success rate: {valid_phrases/total_phrases*100:.1f}%" if total_phrases > 0 else "  Phrase success rate: 0.0%")
        print(f"  Paragraphs with valid phrases: {para_ids_with_valid}")
        print(f"  Paragraph success rate: {para_ids_with_valid/unique_para_ids*100:.1f}%" if unique_para_ids > 0 else "  Paragraph success rate: 0.0%")
        
        if empty_extractions_count > 0:
            print(f"\n📊 Empty Extractions Analysis:")
            print(f"  Total empty extractions: {empty_extractions_count}")
            print(f"  Examples of empty extractions:")
            for i, example in enumerate(empty_extractions_examples, 1):
                print(f"    {i}. {example['para_id']} | {example['section_id']}")
                print(f"       Reason: {example['reason'][:100]}...")
                print(f"       Custom ID: {example['custom_id']}")
        
        # Show sample valid extractions
        valid_results = validation_results[validation_results['is_valid'] == True]
        if not valid_results.empty:
            print(f"\n✅ Valid Extraction Examples (first 3):")
            for idx, row in valid_results.head(3).iterrows():
                print(f"\n  Example {idx + 1}:")
                print(f"    para_id: {row['para_id']}")
                print(f"    section_id: {row['section_id']}")
                print(f"    case_law_excerpt: {row['case_law_excerpt'][:100]}...")
                print(f"    legislation_excerpt: {row['legislation_excerpt'][:100]}...")
                print(f"    confidence: {row['confidence']}")
        
        # Show sample invalid extractions
        invalid_results = validation_results[validation_results['is_valid'] == False]
        if not invalid_results.empty:
            print(f"\n❌ Invalid Extraction Examples (first 3):")
            for idx, row in invalid_results.head(3).iterrows():
                print(f"\n  Example {idx + 1}:")
                print(f"    para_id: {row['para_id']}")
                print(f"    section_id: {row['section_id']}")
                print(f"    case_law_excerpt: {row['case_law_excerpt'][:100]}...")
                print(f"    legislation_excerpt: {row['legislation_excerpt'][:100]}...")
                print(f"    confidence: {row['confidence']}")
                print(f"    reason: {row['reason']}")
        
        # Save validation results
        validation_output = 'data/final_test/final/validation_results.csv'
        validation_results.to_csv(validation_output, index=False)
        print(f"\n💾 Saved validation results to: {validation_output}")
        
        return {
            'total_results_processed': len(results),
            'empty_extractions_count': empty_extractions_count,
            'results_with_phrases': len(results) - empty_extractions_count,
            'total_phrases': total_phrases,
            'valid_phrases': valid_phrases,
            'invalid_phrases': total_phrases - valid_phrases,
            'success_rate': valid_phrases/total_phrases*100 if total_phrases > 0 else 0,
            'paragraphs_with_valid_phrases': para_ids_with_valid,
            'paragraph_success_rate': para_ids_with_valid/unique_para_ids*100 if unique_para_ids > 0 else 0,
            'validation_results': validation_results,
            'empty_extractions_examples': empty_extractions_examples
        }
    
    def _simple_validation_fallback(self):
        """Fallback validation when PhraseValidator fails"""
        print(f"\n📊 SIMPLE VALIDATION FALLBACK")
        print(f"{'='*60}")
        
        total_extractions = 0
        valid_extractions = 0
        invalid_extractions = 0
        
        validation_results = []
        
        for result in self.combined_results:
            para_id = result.get('para_id', '')
            section_id = result.get('section_id', '')
            extracted_phrases = result.get('extracted_phrases', [])
            
            if not extracted_phrases:
                continue
            
            total_extractions += len(extracted_phrases)
            
            # Validate each phrase using simple validation
            for phrase in extracted_phrases:
                is_valid = self.simple_validate_extraction(
                    phrase.get('case_law_excerpt', ''),
                    phrase.get('legislation_excerpt', '')
                )
                
                validation_results.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                    'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                    'confidence': phrase.get('confidence', ''),
                    'is_valid': is_valid
                })
                
                if is_valid:
                    valid_extractions += 1
                else:
                    invalid_extractions += 1
        
        print(f"Total extractions: {total_extractions}")
        print(f"Valid extractions: {valid_extractions}")
        print(f"Invalid extractions: {invalid_extractions}")
        if total_extractions > 0:
            print(f"Validation success rate: {valid_extractions/total_extractions*100:.1f}%")
        
        return {
            'total_phrases': total_extractions,
            'valid_phrases': valid_extractions,
            'invalid_phrases': invalid_extractions,
            'success_rate': valid_extractions/total_extractions*100 if total_extractions > 0 else 0,
            'validation_results': validation_results
        }
    
    def parse_section_id(self, section_id):
        """Parse section_id to extract legislation/act and section"""
        # Example: "id/ukpga/2010/15_section-136"
        # Extract: legislation="ukpga/2010/15", section="136"
        
        if not section_id or not section_id.startswith('id/'):
            return None, None
        
        # Remove 'id/' prefix
        clean_id = section_id[3:]
        
        # Split by '_section-'
        if '_section-' in clean_id:
            parts = clean_id.split('_section-')
            legislation = parts[0]
            section = parts[1]
        else:
            # Fallback: try to extract section number
            legislation = clean_id
            section = None
        
        return legislation, section
    
    def create_legislation_dataframes(self):
        """Create separate DataFrames for top 20 most used legislations"""
        print(f"\n📊 CREATING LEGISLATION DATAFRAMES")
        print(f"{'='*60}")
        
        # Parse section_ids and group by legislation
        legislation_groups = defaultdict(list)
        
        for result in self.combined_results:
            para_id = result.get('para_id', '')
            section_id = result.get('section_id', '')
            extracted_phrases = result.get('extracted_phrases', [])
            
            legislation, section = self.parse_section_id(section_id)
            
            if legislation:
                legislation_groups[legislation].append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'legislation': legislation,
                    'section': section,
                    'extracted_phrases': extracted_phrases,
                    'phrase_count': len(extracted_phrases)
                })
        
        # Count legislation usage and get top 20
        legislation_counts = {leg: len(data) for leg, data in legislation_groups.items()}
        top_legislations = sorted(legislation_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print(f"📈 Top 20 Most Used Legislations:")
        for i, (legislation, count) in enumerate(top_legislations, 1):
            print(f"  {i:2d}. {legislation}: {count:,} extractions")
        
        # Create output directory
        output_dir = Path('data/final_test/final/data_for_verification')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrames only for top 20 legislations
        legislation_dfs = {}
        
        for legislation, count in top_legislations:
            data = legislation_groups[legislation]
            df = pd.DataFrame(data)
            
            # Clean legislation name for filename
            clean_name = legislation.replace('/', '_').replace('\\', '_')
            filename = f"{clean_name}_analysis.csv"
            filepath = output_dir / filename
            
            df.to_csv(filepath, index=False)
            
            legislation_dfs[legislation] = df
            
            print(f"📄 {legislation}: {len(df)} rows -> {filename}")
        
        print(f"\n✅ Created {len(legislation_dfs)} legislation DataFrames (Top 20)")
        print(f"📁 Saved to: {output_dir}")
        
        return legislation_dfs
    
    def generate_research_insights(self):
        """Generate insights suitable for a research paper"""
        print(f"\n📊 RESEARCH INSIGHTS")
        print(f"{'='*60}")
        
        # Load data if not already loaded
        if self.original_data is None:
            self.load_original_data()
        
        # 1. Dataset Statistics
        total_para_ids = len(self.original_data['para_id'].unique())
        total_sections = len(self.original_data)
        avg_sections_per_para = total_sections / total_para_ids
        
        print(f"📈 Dataset Statistics:")
        print(f"  Total unique paragraphs: {total_para_ids:,}")
        print(f"  Total section-paragraph pairs: {total_sections:,}")
        print(f"  Average sections per paragraph: {avg_sections_per_para:.2f}")
        
        # 2. Extraction Performance
        extracted_para_ids = set(r.get('para_id', '') for r in self.combined_results if r.get('para_id'))
        extraction_rate = len(extracted_para_ids) / total_para_ids * 100
        
        print(f"\n📈 Extraction Performance:")
        print(f"  Paragraphs with extractions: {len(extracted_para_ids):,}")
        print(f"  Extraction success rate: {extraction_rate:.1f}%")
        
        # 3. Legislation and Section Statistics
        legislation_counts = Counter()
        section_counts = Counter()
        unique_sections = set()
        
        for result in self.combined_results:
            legislation, section = self.parse_section_id(result.get('section_id', ''))
            if legislation:
                legislation_counts[legislation] += 1
                if section:
                    section_counts[section] += 1
                    unique_sections.add(section)
        
        print(f"\n📈 Legislation Statistics:")
        print(f"  Total unique legislations: {len(legislation_counts)}")
        print(f"  Total unique sections: {len(unique_sections)}")
        print(f"  Total section occurrences: {sum(section_counts.values())}")
        
        print(f"\n📈 Top 10 Most Used Legislations:")
        for legislation, count in legislation_counts.most_common(10):
            print(f"  {legislation}: {count:,} extractions")
        
        print(f"\n📈 Top 10 Most Used Sections:")
        for section, count in section_counts.most_common(10):
            print(f"  {section}: {count:,} extractions")
        
        # 4. Quality Metrics
        total_phrases = sum(len(r.get('extracted_phrases', [])) for r in self.combined_results)
        avg_phrases_per_extraction = total_phrases / len(self.combined_results) if self.combined_results else 0
        
        print(f"\n📈 Quality Metrics:")
        print(f"  Total extracted phrases: {total_phrases:,}")
        print(f"  Average phrases per extraction: {avg_phrases_per_extraction:.2f}")
        
        # 5. Confidence Distribution
        confidence_counts = Counter()
        for result in self.combined_results:
            for phrase in result.get('extracted_phrases', []):
                confidence = phrase.get('confidence', 'Unknown')
                confidence_counts[confidence] += 1
        
        print(f"\n📈 Confidence Distribution:")
        for confidence, count in confidence_counts.items():
            percentage = count / sum(confidence_counts.values()) * 100
            print(f"  {confidence}: {count:,} ({percentage:.1f}%)")
        
        return {
            'total_para_ids': total_para_ids,
            'total_sections': total_sections,
            'avg_sections_per_para': avg_sections_per_para,
            'extraction_rate': extraction_rate,
            'total_phrases': total_phrases,
            'avg_phrases_per_extraction': avg_phrases_per_extraction,
            'legislation_distribution': dict(legislation_counts),
            'confidence_distribution': dict(confidence_counts)
        }

def main():
    """Main analysis function"""
    print("🚀 Starting Comprehensive Dataset Analysis")
    print("="*60)
    
    analyzer = DatasetAnalyzer()
    
    # Step 1: Load and combine all results
    print("\n📂 Step 1: Loading and combining results...")
    analyzer.load_original_data()
    analyzer.combine_all_results()
    analyzer.save_combined_results()
    
    # Step 2: Analyze dataset coverage
    print("\n📊 Step 2: Analyzing dataset coverage...")
    coverage_analysis = analyzer.analyze_dataset_coverage()
    
    # Step 3: Analyze extraction quality
    print("\n🔍 Step 3: Analyzing extraction quality...")
    quality_analysis = analyzer.analyze_extraction_quality()
    
    # Step 4: Create legislation DataFrames
    print("\n📋 Step 4: Creating legislation DataFrames...")
    legislation_dfs = analyzer.create_legislation_dataframes()
    
    # Step 5: Generate research insights
    print("\n📈 Step 5: Generating research insights...")
    research_insights = analyzer.generate_research_insights()
    
    # Step 6: Save comprehensive results
    print("\n💾 Step 6: Saving comprehensive results...")
    results_dir = analyzer.save_comprehensive_results()
    
    print(f"\n✅ Analysis Complete!")
    print(f"📁 Main results saved to: {results_dir}")
    print(f"📊 DataFrames saved to: data/final_test/final/data_for_verification/")
    print(f"📋 Combined results: data/final_test/final/combined_all_results.jsonl")
    
    # Summary of key findings
    print(f"\n📈 KEY FINDINGS:")
    print(f"  📊 Total para_ids processed: {coverage_analysis.get('total_para_ids', 'N/A')}")
    print(f"  ✅ Valid phrases: {quality_analysis.get('valid_phrases', 'N/A')}")
    print(f"  ❌ Invalid phrases: {quality_analysis.get('invalid_phrases', 'N/A')}")
    print(f"  📈 Success rate: {quality_analysis.get('success_rate', 'N/A'):.1f}%")
    print(f"  📋 Top legislations: {len(legislation_dfs)} DataFrames created")

if __name__ == "__main__":
    main() 