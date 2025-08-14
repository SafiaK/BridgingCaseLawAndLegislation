import os
import pickle
from pathlib import Path

def examine_vector_store(pickle_file):
    """
    Examine the contents of a vector store pickle file.
    """
    print(f"\n=== Examining: {pickle_file.name} ===")
    
    try:
        # Load the vector store
        with open(pickle_file, 'rb') as f:
            vector_store = pickle.load(f)
        
        print(f"Vector store type: {type(vector_store)}")
        
        # Try to get some basic info
        try:
            # Get a sample of documents
            sample_docs = vector_store.similarity_search("", k=5)
            print(f"Sample documents found: {len(sample_docs)}")
            
            for i, doc in enumerate(sample_docs):
                print(f"\nDocument {i+1}:")
                print(f"  Content length: {len(doc.page_content)}")
                print(f"  Metadata: {doc.metadata}")
                
                # Check if it has section/schedule
                content_lower = doc.page_content.lower()
                metadata_lower = str(doc.metadata).lower()
                
                has_section = 'section' in content_lower or 'section' in metadata_lower
                has_schedule = 'schedule' in content_lower or 'schedule' in metadata_lower
                
                print(f"  Has 'section': {has_section}")
                print(f"  Has 'schedule': {has_schedule}")
                
        except Exception as e:
            print(f"Error getting sample documents: {e}")
        
        # Try to get total document count
        try:
            all_docs = vector_store.similarity_search("", k=100000)
            print(f"\nTotal documents in store: {len(all_docs)}")
            
            # Count documents with section/schedule
            section_count = 0
            schedule_count = 0
            
            for doc in all_docs:
                content_lower = doc.page_content.lower()
                metadata_lower = str(doc.metadata).lower()
                
                if 'section' in content_lower or 'section' in metadata_lower:
                    section_count += 1
                if 'schedule' in content_lower or 'schedule' in metadata_lower:
                    schedule_count += 1
            
            print(f"Documents with 'section': {section_count}")
            print(f"Documents with 'schedule': {schedule_count}")
            
        except Exception as e:
            print(f"Error getting total document count: {e}")
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def main():
    """
    Examine a few vector store pickle files to understand their structure.
    """
    cache_dir = "../data/final_test/case_csvs/vector_cache"
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # Find pickle files
    pickle_files = list(cache_path.glob("vectorstore_*.pkl"))
    print(f"Found {len(pickle_files)} vector store files")
    
    # Examine first few files
    for i, pickle_file in enumerate(pickle_files[:3]):  # Only examine first 3
        examine_vector_store(pickle_file)
        
        if i < 2:  # Add separator between files
            print("\n" + "="*50)

if __name__ == "__main__":
    main() 