import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
import ast
from difflib import SequenceMatcher
import json
class PhraseValidator:
    """
    Validator for extracted legal phrases.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing punctuation and normalizing whitespace.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity score between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Clean both texts
        clean1 = PhraseValidator.clean_text(text1)
        clean2 = PhraseValidator.clean_text(text2)
        
        if not clean1 or not clean2:
            return 0.0
        
        # Calculate similarity using SequenceMatcher
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity
    
    @staticmethod
    def find_best_match(extracted_text: str, full_text: str, threshold: float = 0.8) -> Tuple[bool, float, str, str]:
        """
        Find the best match for extracted text within full text using similarity scoring.
        
        Args:
            extracted_text (str): Text to find
            full_text (str): Text to search in
            threshold (float): Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            Tuple[bool, float, str, str]: (found, similarity_score, reason, matching_text)
        """
        if not extracted_text or not full_text:
            return False, 0.0, "Empty text provided", ""
        
        # Clean the extracted text
        clean_extracted = PhraseValidator.clean_text(extracted_text)
        if not clean_extracted:
            return False, 0.0, "Extracted text is empty after cleaning", ""
        
        # Clean the full text
        clean_full = PhraseValidator.clean_text(full_text)
        if not clean_full:
            return False, 0.0, "Full text is empty after cleaning", ""
        
        # For exact match, find the original text in the full text
        if clean_extracted in clean_full:
            # Find the actual matching sentence in the original full text
            matching_sentence = PhraseValidator._find_matching_sentence(extracted_text, full_text)
            return True, 1.0, "Exact match found", matching_sentence
        
        # If no exact match, try fuzzy matching
        # Split full text into words and try different window sizes
        words = clean_full.split()
        original_word_matches = list(re.finditer(r'\b\w+\b', full_text))
        
        best_similarity = 0.0
        best_match_original = ""
        
        # Try different window sizes around the expected length
        extracted_words = clean_extracted.split()
        expected_length = len(extracted_words)
        
        for window_size in range(max(1, expected_length - 2), expected_length + 3):
            for i in range(len(words) - window_size + 1):
                window_text = " ".join(words[i:i + window_size])
                similarity = PhraseValidator.calculate_similarity(clean_extracted, window_text)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    # Extract original text window directly
                    if i < len(original_word_matches) and (i + window_size - 1) <= len(original_word_matches):
                        start_pos = original_word_matches[i].start()
                        end_pos = original_word_matches[i + window_size - 1].end()
                        best_match_original = full_text[start_pos:end_pos].strip()
                    else:
                        # Fallback to cleaned words if mapping fails
                        best_match_original = " ".join(words[i:i + window_size])
        
        if best_similarity >= threshold:
            return True, best_similarity, f"Fuzzy match found (similarity: {best_similarity:.2f})", best_match_original
        else:
            return False, best_similarity, f"No match found (best similarity: {best_similarity:.2f}, threshold: {threshold})", best_match_original

    @staticmethod
    def _find_matching_sentence(extracted_text: str, full_text: str) -> str:
        """
        Find the sentence in full_text that contains the extracted_text.
        
        Args:
            extracted_text (str): Text to find
            full_text (str): Full text to search in
            
        Returns:
            str: The sentence containing the extracted text
        """
        import re
        
        # Split full text into sentences
        sentences = re.split(r'[.!?]+', full_text)
        
        # Clean extracted text for comparison
        clean_extracted = PhraseValidator.clean_text(extracted_text)
        
        # Find the sentence that contains the extracted text
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and clean_extracted in PhraseValidator.clean_text(sentence):
                return sentence
        
        # If no sentence found, return the extracted text as is
        return extracted_text

    @staticmethod
    def safe_parse_phrases(phrases_str: str) -> List[Dict]:
        """
        Safely parse phrases from string representation.
        
        Args:
            phrases_str (str): String representation of phrases list
            
        Returns:
            List[Dict]: List of phrase dictionaries
        """
        if pd.isna(phrases_str) or phrases_str == '':
            return []
        
        try:
            if isinstance(phrases_str, str):
                # Handle string representations of lists
                phrases_clean = phrases_str.replace("'", '"').encode().decode('unicode_escape')
                
                # Try to parse as JSON first
                try:
                    import json
                    parsed = json.loads(phrases_clean)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict):
                        return [parsed]
                    else:
                        return [str(parsed)]
                except json.JSONDecodeError:
                    pass
                
                # Try ast.literal_eval as fallback
                if phrases_clean.startswith('['):
                    return ast.literal_eval(phrases_clean)
                else:
                    return [phrases_clean]
            elif isinstance(phrases_str, list):
                return phrases_str
            elif isinstance(phrases_str, dict):
                return [phrases_str]
            else:
                return [str(phrases_str)]
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse phrases_str: {phrases_str[:100]}... Error: {e}")
            return []
    
    @staticmethod
    def debug_parse_phrases(phrases_str: str) -> Dict:
        """
        Debug method to understand what's happening with phrase parsing.
        
        Args:
            phrases_str (str): String representation of phrases
            
        Returns:
            Dict: Debug information
        """
        debug_info = {
            'input_type': type(phrases_str).__name__,
            'input_length': len(str(phrases_str)) if phrases_str else 0,
            'input_preview': str(phrases_str)[:200] if phrases_str else None,
            'is_na': pd.isna(phrases_str),
            'is_empty': phrases_str == '' if phrases_str else True,
            'parsed_result': None,
            'parse_success': False,
            'error': None
        }
        
        if pd.isna(phrases_str) or phrases_str == '':
            debug_info['parsed_result'] = []
            debug_info['parse_success'] = True
            return debug_info
        
        try:
            if isinstance(phrases_str, str):
                # Handle string representations of lists
                phrases_clean = phrases_str.replace("'", '"').encode().decode('unicode_escape')
                debug_info['cleaned_preview'] = phrases_clean[:200]
                
                # Try to parse as JSON first
                try:
                    
                    parsed = json.loads(phrases_clean)
                    if isinstance(parsed, list):
                        debug_info['parsed_result'] = parsed
                        debug_info['parse_success'] = True
                        debug_info['parse_method'] = 'json.loads'
                        return debug_info
                    elif isinstance(parsed, dict):
                        debug_info['parsed_result'] = [parsed]
                        debug_info['parse_success'] = True
                        debug_info['parse_method'] = 'json.loads (dict)'
                        return debug_info
                    else:
                        debug_info['parsed_result'] = [str(parsed)]
                        debug_info['parse_success'] = True
                        debug_info['parse_method'] = 'json.loads (other)'
                        return debug_info
                except json.JSONDecodeError as e:
                    debug_info['json_error'] = str(e)
                
                # Try ast.literal_eval as fallback
                if phrases_clean.startswith('['):
                    parsed = ast.literal_eval(phrases_clean)
                    debug_info['parsed_result'] = parsed
                    debug_info['parse_success'] = True
                    debug_info['parse_method'] = 'ast.literal_eval'
                    return debug_info
                else:
                    debug_info['parsed_result'] = [phrases_clean]
                    debug_info['parse_success'] = True
                    debug_info['parse_method'] = 'string_as_list'
                    return debug_info
            elif isinstance(phrases_str, list):
                debug_info['parsed_result'] = phrases_str
                debug_info['parse_success'] = True
                debug_info['parse_method'] = 'already_list'
                return debug_info
            elif isinstance(phrases_str, dict):
                debug_info['parsed_result'] = [phrases_str]
                debug_info['parse_success'] = True
                debug_info['parse_method'] = 'already_dict'
                return debug_info
            else:
                debug_info['parsed_result'] = [str(phrases_str)]
                debug_info['parse_success'] = True
                debug_info['parse_method'] = 'str_conversion'
                return debug_info
        except (ValueError, SyntaxError) as e:
            debug_info['error'] = str(e)
            debug_info['parsed_result'] = []
            return debug_info
    
    @staticmethod
    def validate_phrase_match(case_law_excerpt: str, legislation_excerpt: str, 
                            paragraph_text: str, section_text: str, 
                            similarity_threshold: float = 0.8) -> Tuple[bool, str, str, str]:
        """
        Validate if extracted phrases actually exist in their respective texts using similarity scoring.
        
        Args:
            case_law_excerpt (str): Extracted phrase from case law
            legislation_excerpt (str): Extracted phrase from legislation
            paragraph_text (str): Full paragraph text
            section_text (str): Full legislation section text
            similarity_threshold (float): Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            Tuple[bool, str, str, str]: (is_valid, reason, actual_case_law_sentence, actual_legislation_sentence)
        """
        # Check if excerpts are provided
        if not case_law_excerpt:
            return False, "No case law excerpt extracted", "", ""
        elif not legislation_excerpt:
            return False, "No legislation excerpt extracted", "", ""
        
        # Find best matches using similarity scoring
        case_law_found, case_law_similarity, case_law_reason, case_law_matching_text = PhraseValidator.find_best_match(
            case_law_excerpt, paragraph_text, similarity_threshold
        )
        
        legislation_found, legislation_similarity, legislation_reason, legislation_matching_text = PhraseValidator.find_best_match(
            legislation_excerpt, section_text, similarity_threshold
        )
        
        # Determine validation result
        if not case_law_found:
            return False, f"Case law excerpt not found: {case_law_reason}", case_law_matching_text, legislation_matching_text
        elif not legislation_found:
            return False, f"Legislation excerpt not found: {legislation_reason}", case_law_matching_text, legislation_matching_text
        elif case_law_found and legislation_found:
            avg_similarity = (case_law_similarity + legislation_similarity) / 2
            return True, f"Both excerpts found (avg similarity: {avg_similarity:.2f})", case_law_matching_text, legislation_matching_text
        else:
            return False, "Unknown validation error", case_law_matching_text, legislation_matching_text
    
    @staticmethod
    def validate_extractions_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate all extractions in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with columns:
                - para_id: Paragraph identifier
                - paragraphs: Paragraph text
                - section_text: Legislation section text
                - section_id: Legislation section identifier
                - extracted_phrases: Extracted phrases (string or list)
                
        Returns:
            pd.DataFrame: DataFrame with validation results
        """
        results = []
        
        for idx, row in df.iterrows():
            para_id = row['para_id']
            paragraph_text = str(row['paragraphs'])
            section_text = str(row['section_text'])
            section_id = str(row.get('section_id', ''))
            
            # Parse extracted phrases
            raw_extracted_phrases = PhraseValidator.safe_parse_phrases(row.get('extracted_phrases', ''))
            
            if not raw_extracted_phrases or not isinstance(raw_extracted_phrases, list):
                results.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'paragraph_text': paragraph_text,
                    'section_text': section_text,
                    'is_valid': False,
                    'reason': 'No extracted phrases found',
                    'case_law_excerpt': None,
                    'legislation_excerpt': None,
                    'confidence': None,
                    'reasoning': None
                })
                continue
            
            # Check each extracted phrase
            found_valid = False
            for phrase_dict in raw_extracted_phrases:
                if not isinstance(phrase_dict, dict):
                    continue
                
                # Get the terms - handle multiple possible field name variations
                case_law_excerpt = (phrase_dict.get('case_law_excerpt') or 
                                  phrase_dict.get('case_law_term') or 
                                  phrase_dict.get('caselaw_term') or
                                  phrase_dict.get('case_law_phrase') or
                                  phrase_dict.get('case_law'))
                
                legislation_excerpt = (phrase_dict.get('legislation_excerpt') or 
                                    phrase_dict.get('legislation_term') or 
                                    phrase_dict.get('legislation_phrase') or
                                    phrase_dict.get('legislation'))
                
                confidence = phrase_dict.get('confidence')
                reasoning = phrase_dict.get('reasoning') or phrase_dict.get('reason')
                
                # Convert to strings
                case_law_excerpt = str(case_law_excerpt) if case_law_excerpt else None
                legislation_excerpt = str(legislation_excerpt) if legislation_excerpt else None
                
                # Validate the phrase match
                is_valid, reason, actual_case_law_sentence, actual_legislation_sentence = PhraseValidator.validate_phrase_match(
                    case_law_excerpt, legislation_excerpt, paragraph_text, section_text
                )
                
                results.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'paragraph_text': paragraph_text,
                    'section_text': section_text,
                    'is_valid': is_valid,
                    'reason': reason,
                    'case_law_excerpt': case_law_excerpt,
                    'legislation_excerpt': legislation_excerpt,
                    'confidence': confidence,
                    'reasoning': reasoning
                })
                
                if is_valid:
                    found_valid = True
                    break  # Found at least one valid extraction
            
            # If no valid extraction found for this paragraph
            if not found_valid and not any(isinstance(phrase_dict, dict) for phrase_dict in raw_extracted_phrases):
                results.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'paragraph_text': paragraph_text,
                    'section_text': section_text,
                    'is_valid': False,
                    'reason': 'No valid phrase dictionaries found',
                    'case_law_excerpt': None,
                    'legislation_excerpt': None,
                    'confidence': None,
                    'reasoning': None
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def get_validation_summary(validation_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics from validation results.
        
        Args:
            validation_df (pd.DataFrame): DataFrame with validation results
            
        Returns:
            Dict: Summary statistics
        """
        total_paragraphs = validation_df['para_id'].nunique()
        valid_paragraphs = validation_df[validation_df['is_valid'] == True]['para_id'].nunique()
        invalid_paragraphs = validation_df[validation_df['is_valid'] == False]['para_id'].nunique()
        
        success_rate = (valid_paragraphs / total_paragraphs * 100) if total_paragraphs > 0 else 0
        
        # Count reasons for failures
        failure_reasons = validation_df[validation_df['is_valid'] == False]['reason'].value_counts()
        
        return {
            'total_paragraphs': total_paragraphs,
            'valid_paragraphs': valid_paragraphs,
            'invalid_paragraphs': invalid_paragraphs,
            'success_rate': success_rate,
            'failure_reasons': failure_reasons.to_dict()
        } 

    @staticmethod
    def validate_batch_output(jsonl_file_path: str, csv_file_path: str) -> pd.DataFrame:
        """
        Parse output JSONL file and validate results against original CSV data.
        
        Args:
            jsonl_file_path (str): Path to the output JSONL file from batch processing
            csv_file_path (str): Path to the original CSV file with paragraphs and section_text
            
        Returns:
            pd.DataFrame: Validation results with detailed information
        """
        import json
        
        # Load original CSV data
        print(f"📁 Loading original CSV data from: {csv_file_path}")
        original_df = pd.read_csv(csv_file_path)
        print(f"   Loaded {len(original_df)} rows from original CSV")
        
        # Parse JSONL output file
        print(f"📁 Parsing JSONL output from: {jsonl_file_path}")
        parsed_results = []
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse the JSONL line
                    batch_result = json.loads(line.strip())
                    
                    # Extract the LLM response
                    if (batch_result.get('response') and 
                        batch_result['response'].get('status_code') == 200 and
                        batch_result['response'].get('body', {}).get('choices')):
                        
                        # Get the assistant's response content
                        choices = batch_result['response']['body']['choices']
                        if choices and len(choices) > 0:
                            assistant_content = choices[0].get('message', {}).get('content', '')
                            
                            # Parse the JSON content from the assistant
                            try:
                                llm_response = json.loads(assistant_content)
                                
                                # Extract key information
                                para_id = llm_response.get('para_id', '')
                                section_id = llm_response.get('section_id', '')
                                extracted_phrases = llm_response.get('extracted_phrases', [])
                                reason = llm_response.get('reason', '')
                                
                                parsed_results.append({
                                    'para_id': para_id,
                                    'section_id': section_id,
                                    'extracted_phrases': extracted_phrases,
                                    'reason': reason,
                                    'line_number': line_num,
                                    'custom_id': batch_result.get('custom_id', ''),
                                    'response_status': batch_result['response'].get('status_code', ''),
                                    'model': batch_result['response'].get('body', {}).get('model', '')
                                })
                                
                            except json.JSONDecodeError as e:
                                print(f"⚠️  Warning: Failed to parse LLM response JSON at line {line_num}: {e}")
                                continue
                        else:
                            print(f"⚠️  Warning: No choices found in response at line {line_num}")
                    else:
                        print(f"⚠️  Warning: Invalid response structure at line {line_num}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing JSONL line {line_num}: {e}")
                    continue
        
        print(f"✅ Parsed {len(parsed_results)} valid results from JSONL")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(parsed_results)
        
        # Merge with original data for validation
        print("🔗 Merging results with original data...")
        
        # Create a merge key from para_id and section_id
        results_df['merge_key'] = results_df['para_id'] + '|' + results_df['section_id']
        original_df['merge_key'] = original_df['para_id'] + '|' + original_df['section_id']
        
        # Merge the data
        merged_df = pd.merge(
            results_df, 
            original_df[['para_id', 'section_id', 'paragraphs', 'section_text', 'merge_key']], 
            on='merge_key', 
            how='left',
            suffixes=('_result', '_original')
        )
        
        print(f"✅ Merged data: {len(merged_df)} rows")
        
        # Validate each result
        print("🔍 Validating extracted phrases...")
        validation_results = []
        
        for idx, row in merged_df.iterrows():
            para_id = row['para_id_result']
            section_id = row['section_id_result']
            paragraph_text = str(row.get('paragraphs', ''))
            section_text = str(row.get('section_text', ''))
            extracted_phrases = row.get('extracted_phrases', [])
            
            # Validate each extracted phrase
            if not extracted_phrases:
                validation_results.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'paragraph_text': paragraph_text,
                    'section_text': section_text,
                    'is_valid': False,
                    'reason': 'No extracted phrases found',
                    'case_law_excerpt': None,
                    'legislation_excerpt': None,
                    'confidence': None,
                    'reasoning': None,
                    'total_phrases': 0,
                    'valid_phrases': 0
                })
                continue
            
            # Validate each phrase in the list
            valid_phrases = 0
            total_phrases = len(extracted_phrases)
            
            for phrase_idx, phrase_dict in enumerate(extracted_phrases):
                if not isinstance(phrase_dict, dict):
                    continue
                
                # Get the terms - handle multiple possible field name variations
                case_law_excerpt = (phrase_dict.get('case_law_excerpt') or 
                                  phrase_dict.get('case_law_term') or 
                                  phrase_dict.get('caselaw_term') or
                                  phrase_dict.get('case_law_phrase') or
                                  phrase_dict.get('case_law') or '')
                
                legislation_excerpt = (phrase_dict.get('legislation_excerpt') or 
                                    phrase_dict.get('legislation_term') or 
                                    phrase_dict.get('legislation_phrase') or
                                    phrase_dict.get('legislation') or '')
                
                confidence = phrase_dict.get('confidence', '')
                reasoning = phrase_dict.get('reasoning', '')
                
                # Validate this specific phrase
                is_valid, reason, actual_case_law_sentence, actual_legislation_sentence = PhraseValidator.validate_phrase_match(
                    case_law_excerpt, legislation_excerpt, paragraph_text, section_text
                )
                
                if is_valid:
                    valid_phrases += 1
                
                validation_results.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'paragraph_text': paragraph_text,
                    'section_text': section_text,
                    'is_valid': is_valid,
                    'reason': reason,
                    'case_law_excerpt': case_law_excerpt,
                    'legislation_excerpt': legislation_excerpt,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'phrase_index': phrase_idx,
                    'total_phrases': total_phrases,
                    'valid_phrases': valid_phrases
                })
        
        # Convert to DataFrame
        validation_df = pd.DataFrame(validation_results)
        
        print(f"✅ Validation complete: {len(validation_df)} phrase validations")
        
        # Debug field names if there are any extracted phrases
        all_extracted_phrases = []
        for _, row in merged_df.iterrows():
            extracted_phrases = row.get('extracted_phrases', [])
            if extracted_phrases:
                all_extracted_phrases.extend(extracted_phrases)
        
        if all_extracted_phrases:
            debug_info = PhraseValidator.debug_field_names(all_extracted_phrases)
            print(f"\n🔍 Field name analysis:")
            print(f"   Total phrases analyzed: {debug_info['total_phrases']}")
            print(f"   Field names found: {debug_info['field_names']}")
            if debug_info['sample_phrase']:
                print(f"   Sample phrase keys: {list(debug_info['sample_phrase'].keys())}")
        
        return validation_df

    @staticmethod
    def get_batch_validation_summary(validation_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for batch validation results.
        
        Args:
            validation_df (pd.DataFrame): Validation results from validate_batch_output
            
        Returns:
            Dict: Summary statistics
        """
        if validation_df.empty:
            return {
                'total_paragraphs': 0,
                'total_phrases': 0,
                'valid_phrases': 0,
                'success_rate': 0.0,
                'paragraphs_with_valid_phrases': 0,
                'paragraph_success_rate': 0.0
            }
        
        # Get unique paragraphs
        unique_paragraphs = validation_df[['para_id', 'section_id']].drop_duplicates()
        total_paragraphs = len(unique_paragraphs)
        
        # Count total and valid phrases
        total_phrases = len(validation_df)
        valid_phrases = validation_df['is_valid'].sum()
        
        # Count paragraphs with at least one valid phrase
        paragraphs_with_valid = validation_df.groupby(['para_id', 'section_id'])['is_valid'].any().sum()
        
        return {
            'total_paragraphs': total_paragraphs,
            'total_phrases': total_phrases,
            'valid_phrases': valid_phrases,
            'success_rate': (valid_phrases / total_phrases * 100) if total_phrases > 0 else 0.0,
            'paragraphs_with_valid_phrases': paragraphs_with_valid,
            'paragraph_success_rate': (paragraphs_with_valid / total_paragraphs * 100) if total_paragraphs > 0 else 0.0
        } 

    @staticmethod
    def debug_field_names(extracted_phrases: List[Dict]) -> Dict:
        """
        Debug function to identify what field names are being used in extracted phrases.
        
        Args:
            extracted_phrases (List[Dict]): List of extracted phrase dictionaries
            
        Returns:
            Dict: Debug information about field names
        """
        field_names = {}
        
        for phrase_dict in extracted_phrases:
            if isinstance(phrase_dict, dict):
                for key in phrase_dict.keys():
                    if key not in field_names:
                        field_names[key] = 0
                    field_names[key] += 1
        
        return {
            'total_phrases': len(extracted_phrases),
            'field_names': field_names,
            'sample_phrase': extracted_phrases[0] if extracted_phrases else None
        } 

    @staticmethod
    def test_similarity_matching():
        """
        Test function to demonstrate similarity matching functionality.
        """
        print("🧪 Testing Similarity Matching")
        print("=" * 50)
        
        # Test cases
        test_cases = [
            {
                "name": "Exact match",
                "extracted": "public interest generally requires",
                "full_text": "The public interest generally requires the precise facts to be recorded.",
                "expected": True
            },
            {
                "name": "Punctuation difference",
                "extracted": "public interest generally requires",
                "full_text": "The public interest, generally, requires the precise facts.",
                "expected": True
            },
            {
                "name": "Case difference",
                "extracted": "Public Interest Generally Requires",
                "full_text": "the public interest generally requires the precise facts",
                "expected": True
            },
            {
                "name": "Minor word difference",
                "extracted": "public interest generally requires",
                "full_text": "The public interest generally requires the precise facts to be recorded.",
                "expected": True
            },
            {
                "name": "No match",
                "extracted": "completely different phrase",
                "full_text": "The public interest generally requires the precise facts.",
                "expected": False
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🔍 Test: {test_case['name']}")
            print(f"   Extracted: '{test_case['extracted']}'")
            print(f"   Full text: '{test_case['full_text']}'")
            
            found, similarity, reason = PhraseValidator.find_best_match(
                test_case['extracted'], test_case['full_text'], threshold=0.8
            )
            
            print(f"   Result: {found} (similarity: {similarity:.2f})")
            print(f"   Reason: {reason}")
            
            if found == test_case['expected']:
                print("   ✅ PASS")
            else:
                print("   ❌ FAIL") 