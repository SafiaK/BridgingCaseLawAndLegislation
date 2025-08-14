import pandas as pd
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

class UtilAnalysis:
    """Utility class for additional analysis functions"""
    
    @staticmethod
    def extract_legislation_info(section_id):
        """Extract detailed legislation information from section_id"""
        if not section_id or not section_id.startswith('id/'):
            return None, None, None, None
        
        # Remove 'id/' prefix
        clean_id = section_id[3:]
        
        # Parse different formats
        # Format 1: ukpga/2010/15_section-136
        if '_section-' in clean_id:
            parts = clean_id.split('_section-')
            legislation = parts[0]
            section = parts[1]
            
            # Extract year and act number
            legislation_parts = legislation.split('/')
            if len(legislation_parts) >= 3:
                act_type = legislation_parts[0]  # ukpga, uksi, etc.
                year = legislation_parts[1]
                act_number = legislation_parts[2]
            else:
                act_type = legislation_parts[0] if legislation_parts else None
                year = None
                act_number = None
                
        else:
            # Fallback parsing
            legislation = clean_id
            section = None
            act_type = None
            year = None
            act_number = None
        
        return {
            'legislation': legislation,
            'section': section,
            'act_type': act_type,
            'year': year,
            'act_number': act_number
        }
    
    @staticmethod
    def analyze_error_patterns(validation_results):
        """Analyze patterns in validation errors"""
        error_patterns = {
            'empty_excerpts': 0,
            'mismatched_lengths': 0,
            'no_similarity': 0,
            'other_errors': 0
        }
        
        for result in validation_results:
            if not result['is_valid']:
                case_law = result['case_law_excerpt']
                legislation = result['legislation_excerpt']
                
                if not case_law or not legislation:
                    error_patterns['empty_excerpts'] += 1
                elif len(case_law) < 10 or len(legislation) < 10:
                    error_patterns['mismatched_lengths'] += 1
                else:
                    error_patterns['no_similarity'] += 1
        
        return error_patterns
    
    @staticmethod
    def create_summary_report(combined_results, original_data):
        """Create a comprehensive summary report"""
        report = {
            'dataset_overview': {},
            'extraction_summary': {},
            'legislation_breakdown': {},
            'quality_metrics': {},
            'error_analysis': {}
        }
        
        # Dataset overview
        unique_para_ids = set(original_data['para_id'].unique())
        extracted_para_ids = set(r.get('para_id', '') for r in combined_results if r.get('para_id'))
        
        report['dataset_overview'] = {
            'total_paragraphs': len(unique_para_ids),
            'total_section_paragraph_pairs': len(original_data),
            'paragraphs_with_extractions': len(extracted_para_ids),
            'extraction_coverage_rate': len(extracted_para_ids) / len(unique_para_ids) * 100
        }
        
        # Extraction summary
        total_extractions = sum(len(r.get('extracted_phrases', [])) for r in combined_results)
        avg_extractions_per_para = total_extractions / len(combined_results) if combined_results else 0
        
        report['extraction_summary'] = {
            'total_extractions': total_extractions,
            'average_extractions_per_paragraph': avg_extractions_per_para,
            'paragraphs_with_extractions': len(extracted_para_ids)
        }
        
        # Legislation breakdown
        legislation_counts = Counter()
        for result in combined_results:
            info = UtilAnalysis.extract_legislation_info(result.get('section_id', ''))
            if info and info['legislation']:
                legislation_counts[info['legislation']] += 1
        
        report['legislation_breakdown'] = dict(legislation_counts.most_common(10))
        
        return report
    
    @staticmethod
    def save_analysis_report(report, output_path):
        """Save analysis report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Analysis report saved to: {output_path}")
    
    @staticmethod
    def create_visualization_data(combined_results):
        """Prepare data for visualizations"""
        viz_data = {
            'legislation_distribution': {},
            'confidence_distribution': {},
            'extraction_timeline': {},
            'quality_metrics': {}
        }
        
        # Legislation distribution
        legislation_counts = Counter()
        for result in combined_results:
            info = UtilAnalysis.extract_legislation_info(result.get('section_id', ''))
            if info and info['legislation']:
                legislation_counts[info['legislation']] += 1
        
        viz_data['legislation_distribution'] = dict(legislation_counts.most_common(10))
        
        # Confidence distribution
        confidence_counts = Counter()
        for result in combined_results:
            for phrase in result.get('extracted_phrases', []):
                confidence = phrase.get('confidence', 'Unknown')
                confidence_counts[confidence] += 1
        
        viz_data['confidence_distribution'] = dict(confidence_counts)
        
        # Quality metrics
        total_phrases = sum(len(r.get('extracted_phrases', [])) for r in combined_results)
        phrases_with_confidence = sum(
            1 for r in combined_results 
            for phrase in r.get('extracted_phrases', [])
            if phrase.get('confidence')
        )
        
        viz_data['quality_metrics'] = {
            'total_phrases': total_phrases,
            'phrases_with_confidence': phrases_with_confidence,
            'confidence_coverage': phrases_with_confidence / total_phrases * 100 if total_phrases > 0 else 0
        }
        
        return viz_data
    
    @staticmethod
    def export_for_research(combined_results, output_dir):
        """Export data in formats suitable for research analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export as CSV for statistical analysis
        research_data = []
        for result in combined_results:
            para_id = result.get('para_id', '')
            section_id = result.get('section_id', '')
            phrases = result.get('extracted_phrases', [])
            
            info = UtilAnalysis.extract_legislation_info(section_id)
            
            for phrase in phrases:
                research_data.append({
                    'para_id': para_id,
                    'section_id': section_id,
                    'legislation': info.get('legislation', '') if info else '',
                    'act_type': info.get('act_type', '') if info else '',
                    'year': info.get('year', '') if info else '',
                    'section': info.get('section', '') if info else '',
                    'case_law_excerpt': phrase.get('case_law_excerpt', ''),
                    'legislation_excerpt': phrase.get('legislation_excerpt', ''),
                    'confidence': phrase.get('confidence', ''),
                    'reasoning': phrase.get('reasoning', ''),
                    'phrase_length': len(phrase.get('case_law_excerpt', ''))
                })
        
        df = pd.DataFrame(research_data)
        csv_path = output_path / 'research_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"📊 Research data exported to: {csv_path}")
        
        return csv_path
    
    @staticmethod
    def generate_statistical_summary(combined_results):
        """Generate statistical summary for research paper"""
        stats = {
            'total_extractions': len(combined_results),
            'total_phrases': sum(len(r.get('extracted_phrases', [])) for r in combined_results),
            'unique_paragraphs': len(set(r.get('para_id', '') for r in combined_results if r.get('para_id'))),
            'unique_sections': len(set(r.get('section_id', '') for r in combined_results if r.get('section_id'))),
            'legislation_coverage': len(set(
                UtilAnalysis.extract_legislation_info(r.get('section_id', '')).get('legislation', '')
                for r in combined_results
                if UtilAnalysis.extract_legislation_info(r.get('section_id', ''))
            )),
            'confidence_distribution': Counter(
                phrase.get('confidence', 'Unknown')
                for r in combined_results
                for phrase in r.get('extracted_phrases', [])
            ),
            'average_phrases_per_extraction': sum(
                len(r.get('extracted_phrases', [])) for r in combined_results
            ) / len(combined_results) if combined_results else 0
        }
        
        return stats 