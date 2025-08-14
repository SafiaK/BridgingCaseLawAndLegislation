import os
import pickle
import shutil
from pathlib import Path

def is_section_or_schedule_document(doc):
    """
    Check if a document has 'section' or 'schedule' in its metadata.
    """
    # Check various metadata fields that might contain section/schedule info
    metadata_fields = ['section_id', 'original_section', 'id', 'file_name', 'source']
    
    for field in metadata_fields:
        value = doc.metadata.get(field, '').lower()
        if 'section' in value or 'schedule' in value:
            return True
    
    # Also check the page content for section/schedule references
    content = doc.page_content.lower()
    if 'section' in content or 'schedule' in content:
        return True
    
    return False

def filter_vector_store_documents(vector_store):
    """
    Filter documents from a vector store to keep only those with section/schedule.
    Returns the filtered documents list.
    """
    try:
        # Get all documents from the vector store
        all_docs = vector_store.similarity_search("", k=100000)  # Get all documents
        
        print(f"  Original documents: {len(all_docs)}")
        
        # Filter documents
        filtered_docs = []
        for doc in all_docs:
            if is_section_or_schedule_document(doc):
                filtered_docs.append(doc)
        
        print(f"  Filtered documents: {len(filtered_docs)}")
        
        return filtered_docs
            
    except Exception as e:
        print(f"  Error filtering vector store: {e}")
        return []

def process_vector_cache_simple(cache_dir, backup=True):
    """
    Process all vector store pickle files in the cache directory.
    This version doesn't require embeddings model initialization.
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # Create backup directory if requested
    if backup:
        backup_dir = cache_path.parent / f"{cache_path.name}_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"Created backup directory: {backup_dir}")
    
    # Find all pickle files
    pickle_files = list(cache_path.glob("vectorstore_*.pkl"))
    print(f"Found {len(pickle_files)} vector store files to process")
    
    processed_count = 0
    filtered_count = 0
    error_count = 0
    removed_count = 0
    
    for pickle_file in pickle_files:
        print(f"\nProcessing: {pickle_file.name}")
        
        try:
            # Load the vector store
            with open(pickle_file, 'rb') as f:
                vector_store = pickle.load(f)
            
            # Filter the documents
            filtered_docs = filter_vector_store_documents(vector_store)
            
            if filtered_docs:
                # Update the vector store's document store with filtered documents
                # This is a bit hacky but should work for FAISS stores
                try:
                    # Clear the existing docstore
                    vector_store.docstore._dict.clear()
                    
                    # Add filtered documents back
                    for i, doc in enumerate(filtered_docs):
                        vector_store.docstore._dict[i] = doc
                    
                    # Save the filtered vector store
                    with open(pickle_file, 'wb') as f:
                        pickle.dump(vector_store, f)
                    
                    processed_count += 1
                    filtered_count += 1
                    print(f"  ✓ Successfully filtered and saved")
                    
                except Exception as e:
                    print(f"  ✗ Error updating vector store: {e}")
                    error_count += 1
            else:
                # If no documents remain, remove the file
                if backup:
                    shutil.move(str(pickle_file), str(backup_dir / pickle_file.name))
                else:
                    pickle_file.unlink()
                print(f"  ✓ Removed file (no section/schedule documents)")
                processed_count += 1
                removed_count += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {pickle_file.name}: {e}")
            error_count += 1
    
    print(f"\n=== Processing Summary ===")
    print(f"Total files: {len(pickle_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Files with filtered content: {filtered_count}")
    print(f"Files removed (no section/schedule): {removed_count}")
    print(f"Errors: {error_count}")

def main():
    """
    Main function to filter vector cache.
    """
    # Configuration
    cache_dir = "../data/final_test/case_csvs/vector_cache"
    
    print("=== Vector Cache Filtering Tool (Simple Version) ===")
    print(f"Cache directory: {cache_dir}")
    print("This will filter all vector store pickle files to keep only documents with 'section' or 'schedule'")
    print("Note: This version doesn't require OpenAI API key setup")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Process the cache
    process_vector_cache_simple(cache_dir, backup=True)
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main() 