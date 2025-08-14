import os
import pickle
import shutil
from pathlib import Path
import json
import hashlib

def get_legislation_hash(legislation_list):
    """
    Generate a hash for a list of legislation IDs.
    This matches the hashing logic used in the original code.
    """
    if not legislation_list:
        return ""
    
    # Sort the legislation list to ensure consistent hashing
    sorted_legislation = sorted(legislation_list)
    legislation_string = "|".join(sorted_legislation)
    return hashlib.md5(legislation_string.encode()).hexdigest()

def is_section_or_schedule_document(doc):
    """
    Check if a document has 'section' or 'schedule' in its metadata ONLY.
    This will remove documents that only have section/schedule in content but not metadata.
    """
    # Check various metadata fields that might contain section/schedule info
    metadata_fields = ['section_id', 'original_section', 'id', 'file_name', 'source']
    
    for field in metadata_fields:
        value = doc.metadata.get(field, '').lower()
        if 'section' in value or 'schedule' in value:
            return True
    
    # Only check metadata, NOT content
    return False

def filter_legislation_vector_store(legislation_dir, backup=True):
    """
    Filter a legislation vector store to keep only documents with section/schedule.
    Returns (original_count, filtered_count, removed_count) or (0, 0, 0) on failure.
    """
    if not legislation_dir.exists():
        print(f"Directory does not exist: {legislation_dir}")
        return 0, 0, 0
    
    pkl_file = legislation_dir / "index.pkl"
    faiss_file = legislation_dir / "index.faiss"
    
    if not pkl_file.exists():
        print(f"No PKL file found in {legislation_dir}")
        return 0, 0, 0
    
    print(f"\n=== Filtering: {legislation_dir.name} ===")
    
    try:
        # Load the vector store
        with open(pkl_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Handle tuple structure
        if isinstance(loaded_data, tuple):
            vector_store = loaded_data[0]
            mapping_dict = loaded_data[1] if len(loaded_data) > 1 else {}
        else:
            vector_store = loaded_data
            mapping_dict = {}
        
        # Get all documents
        if hasattr(vector_store, '_dict'):
            all_docs = list(vector_store._dict.values())
        else:
            print("No documents found in vector store")
            return 0, 0, 0
        
        original_count = len(all_docs)
        print(f"Original documents: {original_count}")
        
        # Filter documents
        filtered_docs = []
        for doc in all_docs:
            if is_section_or_schedule_document(doc):
                filtered_docs.append(doc)
        
        filtered_count = len(filtered_docs)
        removed_count = original_count - filtered_count
        
        print(f"Filtered documents: {filtered_count}")
        print(f"Removed documents: {removed_count}")
        
        if filtered_count == 0:
            print("Warning: No documents remain after filtering!")
            return original_count, 0, original_count
        
        # Create backup if requested
        if backup:
            backup_dir = legislation_dir.parent / f"{legislation_dir.name}_backup"
            if not backup_dir.exists():
                print(f"Creating backup: {backup_dir}")
                shutil.copytree(legislation_dir, backup_dir)
            else:
                print(f"Backup already exists: {backup_dir}")
        
        # Update the vector store with filtered documents
        vector_store._dict.clear()
        for i, doc in enumerate(filtered_docs):
            vector_store._dict[str(i)] = doc
        
        # Update mapping dictionary if it exists
        if mapping_dict:
            new_mapping = {}
            for i, doc in enumerate(filtered_docs):
                new_mapping[i] = str(i)
            mapping_dict = new_mapping
        
        # Save the filtered vector store
        filtered_data = (vector_store, mapping_dict) if mapping_dict else (vector_store,)
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(filtered_data, f)
        
        print(f"Successfully filtered and saved: {pkl_file}")
        return original_count, filtered_count, removed_count
        
    except Exception as e:
        print(f"Error filtering vector store: {e}")
        return 0, 0, 0

def filter_specific_legislation(cache_dir, target_legislation_ids, backup=True):
    """
    Filter specific legislation vector stores.
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # Load processed acts
    processed_acts_file = cache_path / "processed_acts.json"
    if not processed_acts_file.exists():
        print("No processed_acts.json found!")
        return
    
    with open(processed_acts_file, 'r') as f:
        processed_acts = json.load(f)
    
    for legislation_id in target_legislation_ids:
        if legislation_id in processed_acts:
            print(f"\nProcessing: {legislation_id}")
            
            # Calculate hash for the legislation ID
            legislation_hash = hashlib.md5(legislation_id.encode()).hexdigest()
            target_dir = cache_path / legislation_hash
            
            if target_dir.exists():
                success = filter_legislation_vector_store(target_dir, backup=backup)
                if success:
                    print(f"✓ Successfully filtered {legislation_id}")
                else:
                    print(f"✗ Failed to filter {legislation_id}")
            else:
                print(f"✗ Directory not found for {legislation_id} (hash: {legislation_hash})")
        else:
            print(f"✗ {legislation_id} not found in processed acts")

def main():
    """Main function to filter all vector stores in the act embeddings cache."""
    cache_dir = Path("data/final_test/case_csvs/legislation/act_embeddings_cache")
    processed_acts_file = cache_dir / "processed_acts.json"
    
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    if not processed_acts_file.exists():
        print(f"❌ Processed acts file not found: {processed_acts_file}")
        return
    
    # Load all processed acts
    with open(processed_acts_file, 'r') as f:
        processed_acts = json.load(f)
    
    print("=== Filtering All Act Embeddings Cache ===")
    print("This will filter vector stores to keep only documents with 'section' or 'schedule' in metadata")
    print(f"Total acts to process: {len(processed_acts)}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    print()
    
    # Process all legislation IDs
    successful_count = 0
    failed_count = 0
    total_docs_removed = 0
    
    for i, legislation_id in enumerate(processed_acts, 1):
        print(f"Processing {i}/{len(processed_acts)}: {legislation_id}")
        
        try:
            # Calculate hash for this legislation ID
            legislation_hash = get_legislation_hash([legislation_id])
            legislation_dir = cache_dir / legislation_hash
            
            if not legislation_dir.exists():
                print(f"  ⚠️  Directory not found: {legislation_hash}")
                failed_count += 1
                continue
            
            pkl_file = legislation_dir / "index.pkl"
            if not pkl_file.exists():
                print(f"  ⚠️  PKL file not found: {pkl_file}")
                failed_count += 1
                continue
            
            # Filter this vector store
            original_count, filtered_count, removed_count = filter_legislation_vector_store(legislation_dir, backup=True)
            total_docs_removed += removed_count
            
            print(f"  ✓ Successfully filtered {legislation_id}")
            print(f"    Original: {original_count}, Filtered: {filtered_count}, Removed: {removed_count}")
            successful_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {legislation_id}: {str(e)}")
            failed_count += 1
        
        print()
    
    print("=== Filtering Complete ===")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Total documents removed: {total_docs_removed}")
    print("All vector stores have been filtered to keep only documents with 'section' or 'schedule' in metadata.")

if __name__ == "__main__":
    main() 