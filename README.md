# Bridging Case Law and Legislation

A comprehensive system for extracting and analyzing relationships between case law and legislation using advanced NLP techniques and LLM-based extraction.

## 🎯 Project Overview

This project focuses on bridging the gap between case law and legislation by:
- Extracting relevant phrases from case law that reference specific legislation
- Identifying how legislation is applied in judicial decisions
- Creating structured datasets for legal analysis and research

## 🏗️ Architecture

### Core Components

1. **Data Extraction Pipeline**
   - JSONL parsing for multiple LLM outputs (OpenAI, Deepseek, Llama)
   - Robust error handling and failed request tracking
   - Comprehensive status reporting

2. **Text Processing**
   - Special character cleaning and normalization
   - Section identification and validation
   - Dynamic column mapping for source data

3. **Vector Search & Embeddings**
   - FAISS-based similarity search
   - Legislation embedding management
   - Cached vector stores for performance

## 📁 Project Structure

```
BridgingCaseLawAndLegislation/
├── src/                          # Source code
│   ├── extract_data_from_jsonl.py    # Main extraction functions
│   ├── fix_and_convert_jsonl.py      # JSON error fixing
│   ├── ActEmbeddingsManager.py       # Embedding management
│   ├── FAISSManager.py               # FAISS vector database operations
│   ├── relevant_section_finder.py    # Core RAG functions
│   ├── util.py                       # Case law downloading utilities
│   ├── main.py                       # Main pipeline functions
│   ├── experiment_utils.py           # Utility functions for processing
│   ├── LegislationHandler.py         # Legislation parsing and downloading
│   ├── keyPhraseExtractor.py         # Core phrase extraction
│   ├── classifier.py                 # Paragraph classification
│   ├── openAIHandler.py              # OpenAI API integration
│   ├── JudgementHandler.py           # Case law XML processing
│   ├── creat_batch_requests_for_phrase_extraction.py  # Batch creation
│   ├── submit_batch_requests.py      # Batch submission
│   ├── check_batch_status.py         # Batch monitoring
│   ├── make_batch_jsonl_law_application.py  # JSONL batch formatting
│   ├── main_pipeline_Without_phase1.py      # Alternative pipeline
│   ├── LegislationHandler_old.py            # Legacy legislation handler
│   ├── test_combined_map.py                 # Testing utilities
│   ├── create_verification_csv.py           # Verification data creation
│   ├── main_analysis.py                     # Analysis utilities
│   ├── util_analysis.py                     # Analysis helper functions
│   ├── combine_jsonl_files.py               # File combination utilities
│   ├── convert_fixed_to_csv.py             # CSV conversion
│   ├── filter_act_embeddings.py            # Embedding filtering
│   ├── explore_vector_cache.py             # Vector cache exploration
│   ├── examine_vector_cache.py             # Vector cache examination
│   ├── filter_specific_legislation.py      # Legislation filtering
│   ├── filter_vector_cache_simple.py       # Simple cache filtering
│   ├── prepare_training_data.py            # Training data preparation
│   ├── phrase_validator.py                 # Phrase validation
│   └── test.py                             # Testing scripts
├── data/                         # Data files
│   ├── final_test/               # Test datasets
│   │   └── final/reexperiment/fewhot/11August/
│   │       ├── llama_combined_output_final.csv  # Llama extraction results
│   │       └── ...                              # Other model outputs
│   └── newData/                  # New data sources
├── Final_Experimets/             # Analysis notebooks
│   ├── processOpenAI.ipynb       # GPT-4o-mini processing pipeline
│   ├── processingForPhase2.ipynb # Main processing and analysis
│   ├── verifier.ipynb            # Disagreement resolution
│   ├── analysis.ipynb            # Final dataset analysis
│   ├── findingSectionsForMissing.ipynb  # RAG-based section retrieval
│   ├── fewshot_extraction_analysis.ipynb # Complete extraction pipeline
│   ├── make Packages.ipynb       # Evaluation package generation
│   └── case_law_category_analysis.ipynb # Case law category analysis
├── helper_data_files/            # Configuration and examples
│   ├── redo_extraction_prompt.txt        # Main extraction prompt
│   └── phrase_extraction_examples.json   # Curated examples
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

# Code Files in relevance to Paper#
## 1. Downloading the case files

Downloads UK case law data from the National Archives using their Atom XML feed.

**File**: `src/util.py`

**Functions**:
- `fetch_atom_feed(page)` - Fetches Atom feed pages
- `parse_caselaws(tree, START_YEAR, END_YEAR)` - Extracts case URLs by year range
- `download_xml_cases(case_links)` - Downloads XML files to year-based folders
- `fetch_all_pages(start_page, START_YEAR, END_YEAR, end_page)` - Main orchestrator

**Source**: `https://caselaw.nationalarchives.gov.uk/atom.xml`

**Output**: XML files organized in year-based folders under `caselaw/` directory-- Data is not github

## 2. Selecting case laws based on categories

**Functionality**: Creates stratified samples of case laws by court categories for balanced analysis
**File**: `src/case_law_category_analysis.ipynb`
**Main Function**: `stratified_sample_caselaws()`
**Categories**: ewhc, ewca, ukftt, eat, uksc (UK court types)
**Method**: Proportional sampling ensuring minimum 5 files per category
**Output**: Balanced dataset with representative distribution across court types

## 3. Downloading the Legislation acts referenced in each caselaw

**Functionality**: Extracts legislation references from case law CSV files and downloads corresponding legislation acts from legislation.gov.uk
**Files**: 
- `src/main.py` - Main pipeline functions
- `src/experiment_utils.py` - Utility functions for processing
- `src/LegislationHandler.py` - Legislation parsing and downloading
**Main Functions**:
- `extract_legislation_references()` - Extracts legislation URLs from case references
- `downloadThelegislationIfNotExist()` - Downloads legislation XML and saves sections as text files
- `process_legislation_references()` - Orchestrates the complete legislation processing pipeline
**Source**: `https://www.legislation.gov.uk/` (UK legislation database)
**Output**: Legislation sections saved as individual text files organized by act structure

%todo put the name of the file that was generated after this with the selected caselaws

## 4. Processing to classify paragraphs as containing application_of_law 

**Functionality**: Classifies case law paragraphs to identify those containing application of law using multiple LLM models and resolves disagreements
**Files**: 
- `Final_Experimets/processOpenAI.ipynb` - GPT-4o-mini processing pipeline
- `Final_Experimets/processingForPhase2.ipynb` - Main processing and analysis
- `Final_Experimets/verifier.ipynb` - Disagreement resolution and verification
- `Final_Experimets/analysis.ipynb` - Final dataset analysis and statistics
**Models Used**:
- **GPT-4o-mini**: Primary classification of paragraphs for application of law
- **Llama-70b**: Primary classification for comparison and validation
- **Claude**: Disagreement resolution between GPT-4o-mini and Llama outputs
**Process**: 
1. Multiple LLM models classify paragraphs independently
2. Agreement analysis identifies conflicting predictions
3. Claude resolves disagreements to create final annotations
4. Output: Positive examples after agreement resolution with confidence scores
%todo put the name of the file that was generated after this with the positive paragraphs

## 5. RAG-based Section Retrieval and Vector Embedding Management

**Functionality**: Implements Retrieval-Augmented Generation (RAG) system to find relevant legislation sections for case law paragraphs using vector similarity search and efficient embedding management
**Main Notebook**: `Final_Experimets/findingSectionsForMissing.ipynb` *(Suggested rename: `rag_section_retrieval_pipeline.ipynb`)*
**Core Files in `src/`**:
- `ActEmbeddingsManager.py` - Manages legislation embeddings and case-specific vector stores
- `FAISSManager.py` - Handles FAISS vector database operations and batched processing
- `relevant_section_finder.py` - Core RAG functions for section retrieval and similarity search
**Key Functions**:
- `process_dataframe_for_sections()` - Batch processes case law data for section matching
- `build_case_vector_store()` - Creates case-specific vector stores from legislation embeddings
- `BuildVectorDB()` - Constructs vector databases with batched processing for token limits
- `process_dataframe_case_by_case_with_act_manager()` - Processes full dataframe case-by-case with individual vector stores
- `find_sections_for_dataframe_with_case_specific_stores()` - Advanced case-by-case processing with optimized vector store management
**RAG Process**:
1. **Vector Base Creation**: Generates embeddings for legislation sections using batched processing
2. **Embedding Reuse**: Caches embeddings to avoid redundant computation across cases
3. **Case-Specific Vector Stores**: Creates optimized vector stores for each case's relevant legislation
4. **Batched Processing**: Handles large-scale embedding generation within token limits
5. **Section Retrieval**: Uses similarity search to find most relevant legislation sections for each paragraph

**Case-by-Case Processing Approach**:
- **Individual Case Processing**: Each case is processed separately with its own vector store containing only relevant legislation
- **Memory Optimization**: Vector stores are built per-case and can be deleted after processing to save disk space
- **Progress Tracking**: Includes resume capability and progress monitoring for large datasets
- **Error Handling**: Gracefully handles cases with missing legislation or vector store failures
**Output**: Enhanced case law dataset with matched legislation sections and similarity scores

% todo put the csv file name of 9002 paragraphs but with the legislation section as well 
## 6. Phrase Extraction Pipeline using Few-Shot Learning

**Functionality**: Extracts legal phrases and excerpts from case law paragraphs that demonstrate application of legislation using few-shot learning with LLM models
**Main Files**:
- **Prompt File**: `helper_data_files/redo_extraction_prompt.txt` - Main extraction prompt with systematic analysis methodology
- **Few-Shot Examples**: `helper_data_files/phrase_extraction_examples.json` - Curated examples showing case law to legislation phrase matching
**Batch Processing Pipeline** (in `src/` folder):
- `creat_batch_requests_for_phrase_extraction.py` - Creates JSONL batches for phrase extraction requests
- `submit_batch_requests.py` - Submits batches to LLM providers (OpenAI, Anthropic, etc.)
- `check_batch_status.py` - Monitors batch processing status and completion
- `extract_data_from_jsonl.py` - Extracts and processes results from completed batches
**LLM Models Used**:
- **GPT-4o-mini**: 
- **Llama-3.3-70b-versatile**: 
- **DeepSeek-R1-Distill-Llama-70b**: 

**Extraction Process**:
1. **Input**: Case law paragraphs with matched legislation sections (from step 5)
2. **Few-Shot Learning**: Uses curated examples to train the model on extraction patterns
3. **Batch Creation**: Converts dataset into JSONL format for efficient processing
4. **LLM Processing**: Sends batches to models for phrase extraction
5. **Result Extraction**: Processes completed batches to extract matched phrases
**Output**: Structured data showing exact case law excerpts, legislation excerpts, key concepts, reasoning, and confidence scores 

## 6.2 Phrase Extraction Pipeline using Few-Shot Learning (Claude as Adjudicator and Decision Maker)

**Functionality**: Implements a comprehensive phrase extraction analysis pipeline that processes outputs from multiple LLM models, validates extraction quality, and uses Claude to regenerate failed extractions for cases where all models produced low-confidence results
**Main File**: `Final_Experimets/fewshot_extraction_analysis.ipynb` - Complete analysis and adjudication pipeline
**Source Data**: `data/final_test/final/reexperiment/combined_sourcedf_final_rebuild.csv` - 9,002 paragraphs with matched legislation sections
**LLM Models Used**:
- **GPT-4o-mini**: Primary extraction with systematic analysis methodology
- **Llama-3.3-70b-versatile**: Alternative extraction approach for comparison  
- **DeepSeek-R1-Distill-Llama-70b**: Additional extraction perspective
- **Claude**: Regeneration of failed extractions for low-confidence cases

**Extraction Analysis Process**:
1. **Multi-Model Processing**: Loads and filters outputs from all three LLM models
2. **Data Validation**: Implements `check_valid()` function to verify extracted phrases exist in source text
3. **Confidence Standardization**: Normalizes confidence scores across models (High/Medium/Moderate → High, others → Low)
4. **Missing Record Handling**: Creates dummy records for missing (para_id, section_id) combinations to ensure complete coverage
5. **Quality Filtering**: Applies smart filtering to keep High confidence records when available, Low confidence only when no High exists

**Adjudication and Quality Assurance**:
- **Confidence Pattern Analysis**: Categorizes 9,002 paragraphs into confidence patterns:
  - `a_all_high`: 4,631 para_ids (51.4%) - All models agree on High confidence
  - `b_2high_1low`: 2,604 para_ids (28.9%) - Two models High, one Low
  - `c_1high_2low`: 1,229 para_ids (13.7%) - One model High, two Low  
  - `d_all_low`: 538 para_ids (6.0%) - All models Low confidence
- **Claude Regeneration**: Targets the 538 `d_all_low` cases for Claude-based regeneration
- **Success Rate**: Claude successfully regenerates 27 High confidence extractions from previously failed cases

**Output Files Generated**:
- **`df_final_wide_analysis.csv`**: Comprehensive analysis with all model outputs and confidence patterns
- **`df_source_all_low_for_claude.csv`**: 538 low-confidence cases sent to Claude for regeneration
- **`df_source_for_review.csv`**: Remaining cases (538 - 27 = 511) still needing review after Claude regeneration

**Key Features**:
- **Complete Coverage**: Ensures all 9,002 paragraphs are processed and analyzed
- **Smart Filtering**: Prioritizes High confidence extractions while maintaining Low confidence as fallback
- **Validation Pipeline**: Verifies extracted phrases exist in source text using text matching
- **Targeted Regeneration**: Uses Claude specifically for the most challenging cases
- **Comprehensive Analysis**: Provides detailed breakdown of model agreement patterns and success rates

## 7. Packages for Validation

**Functionality**: Creates evaluation packages and scoring systems for validating the quality of legal phrase extraction results from LLM models
**Main File**: `Final_Experimets/make Packages.ipynb` - Evaluation package generation and scoring notebook
**Purpose**: Generates structured evaluation files and calculates accuracy scores for extraction quality assessment

1. **Sample Evaluation Files**: Creates formatted text files for human evaluation of extraction quality
2. **Failed Cases Analysis**: Generates evaluation packages for cases where all LLM models produced low confidence
3. **Structured Format**: Each evaluation record includes:
   - **paralink**: Direct link to case law paragraph
   - **sectionlink**: Direct link to legislation section
   - **paragraph**: Full paragraph text for review
   - **section**: Full legislation section text
   - **case_law_excerpt**: Extracted case law phrases
   - **legislation_excerpt**: Extracted legislation phrases
   - **reasoning**: Model's reasoning for extraction
   - **Evaluation fields**: YES/NO questions for quality assessment

**Scoring System**:
- **Extraction Accuracy**: `IS_Extracton_Correct` - Measures if extracted phrases are accurate
- **Reasoning Quality**: `IS_Reasoning_Correct` - Evaluates if model reasoning is sound
- **Failure Analysis**: Three failure reason categories:
  - **ReasonA**: Paragraph doesn't contain "application of law"
  - **ReasonB**: Section is irrelevant or not applicable to paragraph
  - **ReasonC**: System failed to extract terms with high confidence

**Output Files**:
- **Sample evaluation files**: For quality assessment of successful extractions
- **Failed cases evaluation**: For analysis of low-confidence extraction failures
- **Scoring results**: Quantitative accuracy metrics and failure reason breakdowns

**Usage**: Run after phrase extraction to generate evaluation packages for human reviewers and calculate quality metrics for the extraction pipeline 



## 8. Files that are on One Drive
- The list of caselaws with proper paragraph ids that were considered as a base for the experiments = - - - proper_case_law_by_court.csv 
- The selected caselaw paragraphs = final_data_for_processing.csv
- The out put of gopt-40-mini = combined_openai_output
- The output of llama = combined_llama_output
- Cases with the agreement = agreement.csv
- Cases with disagreement = disagreement.csv
- Final Positive Paragraphs for Processing for Phase 2 = positve_cases.csv
- The file after attaching 2 sections with each of the paragraph = combined_two_sections_with_paragraphs.csv
- The output csv file from the combined extraction result of gpt-4o-mini = gpt-extract.csv
- The output csv file from the combined extraction result of llama = llama_combined_output_final.csv
- The output csv file from the combined extraction result of deepseek = 
- The data for the failed instances sent to claude for regenration 
- The data sent to the claude for the decision
- Final Dataframe with approved pairs
