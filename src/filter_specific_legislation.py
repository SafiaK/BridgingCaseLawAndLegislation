import os
import pickle
import shutil
from pathlib import Path

def find_vector_stores_with_legislation(cache_dir, target_legislation_ids):
    """
    Find vector stores that contain documents with specific legislation IDs.
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return []
    
    matching_files = []
    pickle_files = list(cache_path.glob("vectorstore_*.pkl"))
    
    print(f"Searching through {len(pickle_files)} vector store files...")
    
    for pickle_file in pickle_files:
        print(f"\nChecking: {pickle_file.name}")
        
        try:
            # Load the vector store
            with open(pickle_file, 'rb') as f:
                vector_store = pickle.load(f)
            
            # Get a sample of documents to check for legislation IDs
            try:
                sample_docs = vector_store.similarity_search("", k=10)
                
                found_legislation = set()
                for doc in sample_docs:
                    legislation_id = doc.metadata.get('legislation_id', '')
                    if legislation_id:
                        found_legislation.add(legislation_id)
                
                # Check if any of our target legislation IDs are in this store
                matching_ids = found_legislation.intersection(set(target_legislation_ids))
                
                if matching_ids:
                    print(f"  ✓ Found legislation IDs: {matching_ids}")
                    matching_files.append((pickle_file, matching_ids))
                else:
                    print(f"  - No target legislation found")
                    
            except Exception as e:
                print(f"  ✗ Error checking documents: {e}")
                
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
    
    return matching_files

def is_section_or_schedule_document(doc):
    """
    Check if a document has 'section' or 'schedule' in its metadata or content.
    """
    # Check metadata fields
    metadata_fields = ['section_id', 'original_section', 'id', 'file_name', 'source']
    
    for field in metadata_fields:
        value = doc.metadata.get(field, '').lower()
        if 'section' in value or 'schedule' in value:
            return True
    
    # Check page content
    content = doc.page_content.lower()
    if 'section' in content or 'schedule' in content:
        return True
    
    return False

def filter_specific_vector_store(pickle_file, target_legislation_ids, backup=True):
    """
    Filter a specific vector store to keep only documents with section/schedule.
    """
    print(f"\n=== Filtering: {pickle_file.name} ===")
    
    # Create backup if requested
    if backup:
        backup_dir = pickle_file.parent / f"{pickle_file.parent.name}_backup"
        backup_dir.mkdir(exist_ok=True)
        backup_file = backup_dir / pickle_file.name
    
    try:
        # Load the vector store
        with open(pickle_file, 'rb') as f:
            vector_store = pickle.load(f)
        
        # Get all documents
        all_docs = vector_store.similarity_search("", k=100000)
        print(f"Original documents: {len(all_docs)}")
        
        # Filter documents
        filtered_docs = []
        section_count = 0
        schedule_count = 0
        
        for doc in all_docs:
            # Check if document has section/schedule
            if is_section_or_schedule_document(doc):
                filtered_docs.append(doc)
                
                # Count section vs schedule
                content_lower = doc.page_content.lower()
                metadata_lower = str(doc.metadata).lower()
                
                if 'section' in content_lower or 'section' in metadata_lower:
                    section_count += 1
                if 'schedule' in content_lower or 'schedule' in metadata_lower:
                    schedule_count += 1
        
        print(f"Filtered documents: {len(filtered_docs)}")
        print(f"  - With 'section': {section_count}")
        print(f"  - With 'schedule': {schedule_count}")
        
        if filtered_docs:
            # Create backup
            if backup:
                shutil.copy2(pickle_file, backup_file)
                print(f"Created backup: {backup_file}")
            
            # Update the vector store's document store
            try:
                # Clear existing docstore
                vector_store.docstore._dict.clear()
                
                # Add filtered documents back
                for i, doc in enumerate(filtered_docs):
                    vector_store.docstore._dict[i] = doc
                
                # Save the filtered vector store
                with open(pickle_file, 'wb') as f:
                    pickle.dump(vector_store, f)
                
                print(f"✓ Successfully filtered and saved")
                return True
                
            except Exception as e:
                print(f"✗ Error updating vector store: {e}")
                return False
        else:
            print("No section/schedule documents found - removing file")
            if backup:
                shutil.move(str(pickle_file), str(backup_file))
            else:
                pickle_file.unlink()
            return True
            
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        return False

def main():
    """
    Main function to filter specific legislation vector stores.
    """
    # Configuration
    cache_dir = "../data/final_test/case_csvs/vector_cache"
    target_legislation_ids = [
        'id/ukpga/2010/15',
        'id/ukpga/1996/18'
    ]
    
    print("=== Specific Legislation Vector Store Filter ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Target legislation IDs: {target_legislation_ids}")
    print("This will find and filter vector stores containing these legislation IDs")
    
    # Find matching vector stores
    matching_files = find_vector_stores_with_legislation(cache_dir, target_legislation_ids)
    
    if not matching_files:
        print("\nNo vector stores found with the specified legislation IDs.")
        return
    
    print(f"\nFound {len(matching_files)} vector store(s) with target legislation:")
    for file_path, legislation_ids in matching_files:
        print(f"  - {file_path.name}: {legislation_ids}")
    
    # Ask for confirmation
    response = input("\nDo you want to filter these files? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Filter each matching file
    success_count = 0
    for file_path, legislation_ids in matching_files:
        success = filter_specific_vector_store(file_path, legislation_ids, backup=True)
        if success:
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {len(matching_files)}")
    print(f"Successfully filtered: {success_count}")

if __name__ == "__main__":
    main() 