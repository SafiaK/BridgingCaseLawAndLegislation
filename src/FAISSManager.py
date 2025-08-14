import os
import time
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

class FAISSManager:
    """
    Manager class for handling FAISS vector stores with persistence capabilities.
    Supports creating, loading, saving, and updating indexes.
    """
    
    def __init__(self, cache_dir: str = "cache/faiss_indexes"):
        """
        Initialize the FAISS manager.
        
        Args:
            cache_dir: Directory to store cached FAISS indexes
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, index_name: str) -> str:
        """Get the file path for a cached index"""
        return os.path.join(self.cache_dir, f"{index_name}.pickle")
    
    def index_exists(self, index_name: str) -> bool:
        """Check if a cached index exists"""
        return os.path.exists(self.get_cache_path(index_name))
    
    def load_index(self, index_name: str) -> Optional[FAISS]:
        """
        Load a FAISS index from cache if it exists.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            FAISS index if found, None otherwise
        """
        cache_path = self.get_cache_path(index_name)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, "rb") as f:
                print(f"Loading cached FAISS index from {cache_path}")
                # Deserialize the stored index and embeddings
                stored_data = pickle.load(f)
                
                # Recreate the embeddings instance
                embeddings = OpenAIEmbeddings()
                
                # Load the vectorstore with the embeddings instance
                vectorstore = FAISS.deserialize_from_bytes(
                    serialized=stored_data["index_data"],
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Store metadata about the index if it exists
                if "metadata" in stored_data:
                    vectorstore.index_metadata = stored_data["metadata"]
                else:
                    vectorstore.index_metadata = {"document_count": vectorstore.index.ntotal}
                
                return vectorstore
        except Exception as e:
            print(f"Error loading cached index {index_name}: {str(e)}")
            return None
    
    def save_index(self, vectorstore: FAISS, index_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a FAISS index to cache.
        
        Args:
            vectorstore: The FAISS vector store to save
            index_name: Name for the cached index
            metadata: Optional metadata to store with the index
        """
        cache_path = self.get_cache_path(index_name)
        
        try:
            # Serialize the index
            index_data = vectorstore.serialize_to_bytes()
            
            # Store the serialized data
            if not metadata and hasattr(vectorstore, 'index_metadata'):
                metadata = vectorstore.index_metadata
            
            # Update metadata with current document count
            if not metadata:
                metadata = {}
            metadata["document_count"] = vectorstore.index.ntotal
            metadata["last_updated"] = time.time()
            
            with open(cache_path, "wb") as f:
                pickle.dump({"index_data": index_data, "metadata": metadata}, f)
                
            print(f"Saved FAISS index to {cache_path} with {metadata['document_count']} documents")
        except Exception as e:
            print(f"Error saving index {index_name}: {str(e)}")
            
    def update_index(self, 
                    index_name: str, 
                    new_docs: List[Document],
                    batch_size: int = 50) -> Optional[FAISS]:
        """
        Update an existing index with new documents.
        
        Args:
            index_name: Name of the index to update
            new_docs: New documents to add to the index
            batch_size: Batch size for processing embeddings
            
        Returns:
            Updated FAISS index if successful, None otherwise
        """
        # First load the existing index
        vectorstore = self.load_index(index_name)
        if not vectorstore:
            print(f"Cannot update index {index_name} - not found in cache")
            return None
            
        if not new_docs:
            print("No new documents to add")
            return vectorstore
            
        print(f"Updating index {index_name} with {len(new_docs)} new documents")
        
        try:
            # Process new documents in batches to avoid token limits
            embeddings = vectorstore._embedding
            
            for i in range(0, len(new_docs), batch_size):
                batch = new_docs[i:i + batch_size]
                print(f"Adding batch {i//batch_size + 1}/{(len(new_docs)-1)//batch_size + 1} ({len(batch)} docs)")
                
                try:
                    # Add the documents to the existing store
                    vectorstore.add_documents(batch)
                    print(f"Successfully added batch {i//batch_size + 1}")
                except Exception as e:
                    if "max_tokens_per_request" in str(e):
                        # If batch too large, process one at a time
                        print(f"Batch too large, processing documents individually")
                        for doc in batch:
                            try:
                                vectorstore.add_documents([doc])
                            except Exception as e2:
                                print(f"Failed to add document: {str(e2)}")
                    else:
                        print(f"Error adding batch: {str(e)}")
            
            # Save the updated index
            self.save_index(vectorstore, index_name)
            return vectorstore
            
        except Exception as e:
            print(f"Error updating index {index_name}: {str(e)}")
            return None

def build_or_load_vector_db(
    directory: str, 
    legislation_list: List[str],
    index_name: str = "legislation_index",
    rebuild: bool = False
) -> FAISS:
    """
    Build a vector database from legislation documents or load from cache if available.
    
    Args:
        directory: Directory containing legislation files
        legislation_list: List of legislation numbers to process
        index_name: Name for the cached index
        rebuild: If True, rebuild the index even if cached version exists
        
    Returns:
        FAISS vector store
    """
    # Initialize the FAISS manager
    faiss_manager = FAISSManager()
    
    # Check if we have a cached version and should use it
    if not rebuild and faiss_manager.index_exists(index_name):
        vectorstore = faiss_manager.load_index(index_name)
        if vectorstore:
            return vectorstore
    
    # If we need to build the index
    print(f"Building new FAISS index '{index_name}'...")
    vectorstore = BuildVectorDB(directory, legislation_list)
    
    # Cache the result
    faiss_manager.save_index(vectorstore, index_name)
    
    return vectorstore

def BuildVectorDB(directory, legislation_list):
    """
    Build a vector database from legislation documents with batched processing
    to handle token limit constraints.
    """
    def load_legislative_sections(directory, legislation_number):
        sections = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                try:
                    section_number = filename.split('-')[1].split('.')[0]  # Extract section number
                    with open(os.path.join(directory, filename), 'r') as file:
                        text = file.read().strip()  # Read the content of the file
                        sections.append({
                            "id": f"{legislation_number}_section_{section_number}",
                            "text": text,
                            "legislation_id": legislation_number
                        })
                except:
                    pass
        return sections

    docs = []
    for legislation_number in legislation_list:
        try:
            legislative_sections = load_legislative_sections(
                f"{directory}/{legislation_number}", legislation_number
            )
            doc = [
                Document(page_content=sec["text"], 
                         metadata={
                             "id": sec["id"],
                             "legislation_id": sec["legislation_id"]
                         }) 
                for sec in legislative_sections
            ]
            docs.extend(doc)
        except Exception as e:
            print(f"Error processing legislation {legislation_number}: {str(e)}")
    
    # Now use batched processing to create embeddings
    try:
        vectorstore = build_vector_db_with_batching(docs)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def build_vector_db_with_batching(docs: List[Document], 
                                  batch_size: int = 100, 
                                  max_retries: int = 5,
                                  retry_sleep: int = 2) -> FAISS:
    """
    Build a FAISS vector store from documents using batched embedding to avoid token limits.
    
    Args:
        docs: List of documents to embed
        batch_size: Initial batch size (number of documents)
        max_retries: Maximum number of retries for each batch
        retry_sleep: Seconds to sleep between retries
    
    Returns:
        FAISS vector store with embedded documents
    """
    embeddings = OpenAIEmbeddings()
    
    # Track batches of documents and their embeddings
    all_embeddings = []
    all_texts = []
    all_metadatas = []
    
    # Process documents in batches
    i = 0
    current_batch_size = batch_size
    
    while i < len(docs):
        end_idx = min(i + current_batch_size, len(docs))
        batch = docs[i:end_idx]
        
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]
        
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                # Try to embed the current batch
                print(f"Processing batch {i} to {end_idx} ({len(batch_texts)} documents)")
                batch_embeddings = embeddings.embed_documents(batch_texts)
                
                # If successful, add to our collections
                all_embeddings.extend(batch_embeddings)
                all_texts.extend(batch_texts)
                all_metadatas.extend(batch_metadatas)
                
                print(f"Successfully embedded batch {i} to {end_idx} of {len(docs)}")
                success = True
                
            except Exception as e:
                error_msg = str(e)
                retry_count += 1
                
                if "max_tokens_per_request" in error_msg:
                    # If we hit token limit, reduce batch size by half
                    current_batch_size = max(1, current_batch_size // 2)
                    end_idx = min(i + current_batch_size, len(docs))
                    batch = docs[i:end_idx]
                    batch_texts = [doc.page_content for doc in batch]
                    batch_metadatas = [doc.metadata for doc in batch]
                    
                    print(f"Token limit exceeded. Reducing batch size to {current_batch_size} and retrying...")
                else:
                    print(f"Error in batch {i} to {end_idx}: {error_msg}")
                    print(f"Retry {retry_count}/{max_retries}...")
                
                # Sleep to avoid rate limits
                time.sleep(retry_sleep)
        
        if success:
            i = end_idx  # Move to next batch
        else:
            # If we've exhausted retries, skip this problematic batch
            print(f"Failed to process batch after {max_retries} retries. Skipping to next batch.")
            i = end_idx
    
    # Create FAISS index from collected embeddings
    if not all_embeddings:
        raise ValueError("No documents were successfully embedded")
    
    print(f"Creating FAISS index with {len(all_embeddings)} embeddings")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_embeddings)),
        embedding=embeddings,
        metadatas=all_metadatas
    )
    
    return vector_store

# Example usage
if __name__ == "__main__":
    legislation_dir = "data/legislation"
    acts = ["act1", "act2", "act3"]  # Your list of legislation IDs
    
    # First run will build and cache the index
    vectorstore = build_or_load_vector_db(legislation_dir, acts)
    
    # Subsequent runs will load from cache
    # vectorstore = build_or_load_vector_db(legislation_dir, acts)
    
    # Force rebuild if needed
    # vectorstore = build_or_load_vector_db(legislation_dir, acts, rebuild=True)