import os
import pickle
import hashlib
import json
from pathlib import Path
from collections import defaultdict

def get_legislation_hash(legislation_list):
    """Create a hash for a set of legislation acts - same as BulkCachedProcessor"""
    if legislation_list is None:
        return "no_legislation"
    
    if not isinstance(legislation_list, (list, tuple)):
        return "invalid_legislation"
        
    if len(legislation_list) == 0:
        return "empty_legislation"
        
    sorted_legislation = sorted(legislation_list)
    legislation_string = "|".join(sorted_legislation)
    return hashlib.md5(legislation_string.encode()).hexdigest()

def find_legislation_directory(cache_dir, target_legislation_id):
    """
    Find the directory for a specific legislation ID in the act_embeddings_cache.
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return None
    
    # Load processed acts to see what's available
    processed_acts_file = cache_path / "processed_acts.json"
    if processed_acts_file.exists():
        with open(processed_acts_file, 'r') as f:
            processed_acts = json.load(f)
        print(f"Found {len(processed_acts)} processed acts")
        
        if target_legislation_id in processed_acts:
            print(f"✓ Target legislation '{target_legislation_id}' found in processed acts")
        else:
            print(f"✗ Target legislation '{target_legislation_id}' NOT found in processed acts")
            return None
    
    # Calculate hash for the single legislation ID
    legislation_hash = get_legislation_hash([target_legislation_id])
    target_dir = cache_path / legislation_hash
    
    if target_dir.exists():
        print(f"✓ Found directory for {target_legislation_id}: {target_dir}")
        return target_dir
    else:
        print(f"✗ Directory not found for {target_legislation_id} (hash: {legislation_hash})")
        return None

def examine_legislation_vector_store(legislation_dir):
    """
    Examine the vector store for a specific legislation.
    """
    if not legislation_dir.exists():
        print(f"Directory does not exist: {legislation_dir}")
        return
    
    pkl_file = legislation_dir / "index.pkl"
    faiss_file = legislation_dir / "index.faiss"
    
    print(f"\n=== Examining: {legislation_dir.name} ===")
    print(f"PKL file exists: {pkl_file.exists()}")
    print(f"FAISS file exists: {faiss_file.exists()}")
    
    if not pkl_file.exists():
        print("No PKL file found!")
        return
    
    try:
        # Load the vector store
        with open(pkl_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"Loaded data type: {type(loaded_data)}")
        
        # Handle tuple structure (vector store + metadata)
        if isinstance(loaded_data, tuple):
            print(f"Tuple length: {len(loaded_data)}")
            vector_store = loaded_data[0]  # First element should be the vector store
            print(f"Vector store type: {type(vector_store)}")
            
            # If there are additional elements in the tuple, show them
            if len(loaded_data) > 1:
                print(f"Additional tuple elements:")
                for i, element in enumerate(loaded_data[1:], 1):
                    print(f"  Element {i}: {type(element)} - {len(element) if isinstance(element, dict) else element}")
        else:
            vector_store = loaded_data
            print(f"Vector store type: {type(vector_store)}")
        
        # Access documents from the docstore
        # Check if vector_store itself is a docstore
        if hasattr(vector_store, '_dict'):
            all_docs = list(vector_store._dict.values())
            print(f"Total documents in store: {len(all_docs)}")
            
            # Sample some documents
            sample_size = min(5, len(all_docs))
            sample_docs = all_docs[:sample_size]
            
            section_count = 0
            schedule_count = 0
            other_count = 0
            
            for i, doc in enumerate(sample_docs):
                print(f"\nDocument {i+1}:")
                print(f"  Content preview: {doc.page_content[:100]}...")
                print(f"  Metadata: {doc.metadata}")
                
                # Check if it's a section or schedule
                is_section = any('section' in str(v).lower() for v in doc.metadata.values())
                is_schedule = any('schedule' in str(v).lower() for v in doc.metadata.values())
                
                if is_section:
                    section_count += 1
                elif is_schedule:
                    schedule_count += 1
                else:
                    other_count += 1
            
            print(f"\nDocument types in sample:")
            print(f"  Sections: {section_count}")
            print(f"  Schedules: {schedule_count}")
            print(f"  Other: {other_count}")
            
            # Check all documents for section/schedule
            total_section = 0
            total_schedule = 0
            total_other = 0
            
            for doc in all_docs:
                is_section = any('section' in str(v).lower() for v in doc.metadata.values())
                is_schedule = any('schedule' in str(v).lower() for v in doc.metadata.values())
                
                if is_section:
                    total_section += 1
                elif is_schedule:
                    total_schedule += 1
                else:
                    total_other += 1
            
            print(f"\nTotal document types:")
            print(f"  Sections: {total_section}")
            print(f"  Schedules: {total_schedule}")
            print(f"  Other: {total_other}")
            
        # Check if vector_store has a docstore attribute
        elif hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
            all_docs = list(vector_store.docstore._dict.values())
            print(f"Total documents in store: {len(all_docs)}")
            
            # Sample some documents
            sample_size = min(5, len(all_docs))
            sample_docs = all_docs[:sample_size]
            
            section_count = 0
            schedule_count = 0
            other_count = 0
            
            for i, doc in enumerate(sample_docs):
                print(f"\nDocument {i+1}:")
                print(f"  Content preview: {doc.page_content[:100]}...")
                print(f"  Metadata: {doc.metadata}")
                
                # Check if it's a section or schedule
                is_section = any('section' in str(v).lower() for v in doc.metadata.values())
                is_schedule = any('schedule' in str(v).lower() for v in doc.metadata.values())
                
                if is_section:
                    section_count += 1
                elif is_schedule:
                    schedule_count += 1
                else:
                    other_count += 1
            
            print(f"\nDocument types in sample:")
            print(f"  Sections: {section_count}")
            print(f"  Schedules: {schedule_count}")
            print(f"  Other: {other_count}")
            
            # Check all documents for section/schedule
            total_section = 0
            total_schedule = 0
            total_other = 0
            
            for doc in all_docs:
                is_section = any('section' in str(v).lower() for v in doc.metadata.values())
                is_schedule = any('schedule' in str(v).lower() for v in doc.metadata.values())
                
                if is_section:
                    total_section += 1
                elif is_schedule:
                    total_schedule += 1
                else:
                    total_other += 1
            
            print(f"\nTotal document types:")
            print(f"  Sections: {total_section}")
            print(f"  Schedules: {total_schedule}")
            print(f"  Other: {total_other}")
            
        else:
            print("No docstore found or docstore doesn't have _dict attribute")
            print(f"Available attributes: {dir(vector_store)}")
            
    except Exception as e:
        print(f"Error loading vector store: {e}")

def explore_act_embeddings_cache(cache_dir, target_legislation_ids=None):
    """
    Explore the act embeddings cache to see what legislation IDs are available.
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # Load processed acts
    processed_acts_file = cache_path / "processed_acts.json"
    if processed_acts_file.exists():
        with open(processed_acts_file, 'r') as f:
            processed_acts = json.load(f)
        print(f"Found {len(processed_acts)} processed acts")
        
        # Show first 10 acts
        print("\nFirst 10 processed acts:")
        for i, act in enumerate(processed_acts[:10]):
            print(f"  {i+1}. {act}")
        
        if target_legislation_ids:
            print(f"\nChecking target legislation IDs:")
            for legislation_id in target_legislation_ids:
                if legislation_id in processed_acts:
                    print(f"  ✓ {legislation_id} - Found in processed acts")
                    
                    # Find the directory
                    legislation_hash = get_legislation_hash([legislation_id])
                    target_dir = cache_path / legislation_hash
                    
                    if target_dir.exists():
                        print(f"    Directory: {target_dir}")
                        examine_legislation_vector_store(target_dir)
                    else:
                        print(f"    ✗ Directory not found (hash: {legislation_hash})")
                else:
                    print(f"  ✗ {legislation_id} - NOT found in processed acts")
    else:
        print("No processed_acts.json found!")

def main():
    # Configuration
    cache_dir = "data/final_test/case_csvs/legislation/act_embeddings_cache"
    target_legislation_ids = [
        "id/ukpga/2010/15",
        "id/ukpga/1996/18"
    ]
    
    print("=== Exploring Act Embeddings Cache ===")
    explore_act_embeddings_cache(cache_dir, target_legislation_ids)

if __name__ == "__main__":
    main() 