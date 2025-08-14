#!/usr/bin/env python3
"""
Act Embeddings Manager Module

This module manages act embeddings separately and builds vector stores per case 
using only the relevant acts for that case.

Key Features:
1. Separate Act Embeddings: Each act's embeddings are stored separately
2. Case-Specific Vector Stores: Build vector stores using only acts relevant to each case
3. Efficient Processing: Avoid rebuilding embeddings for acts that have already been processed
4. Incremental Updates: Add new acts to the embedding database as needed
"""

import os
import shutil
import pickle
import hashlib
import faiss

import json
from pathlib import Path
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openAIHandler


class ActEmbeddingsManager:
    """Manages act embeddings separately and builds case-specific vector stores."""
    
    def __init__(self, legislation_folder_path, embeddings_cache_dir, vector_stores_dir, embeddings_model):
        """
        Initialize the ActEmbeddingsManager.
        
        Args:
            legislation_folder_path (str): Path to the legislation folder
            embeddings_cache_dir (str): Directory to store act embeddings
            vector_stores_dir (str): Directory to store case vector stores
            embeddings_model: The embeddings model to use
        """
        self.legislation_folder_path = legislation_folder_path
        self.embeddings_cache_dir = embeddings_cache_dir
        self.vector_stores_dir = vector_stores_dir
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=len,
        )
        
        # Track which acts have been processed
        self.processed_acts_file = os.path.join(embeddings_cache_dir, 'processed_acts.json')
        self.processed_acts = self._load_processed_acts()
        
        # Create directories if they don't exist
        os.makedirs(embeddings_cache_dir, exist_ok=True)
        os.makedirs(vector_stores_dir, exist_ok=True)
    
    def _load_processed_acts(self):
        """Load the list of processed acts from file."""
        if os.path.exists(self.processed_acts_file):
            try:
                with open(self.processed_acts_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading processed acts: {e}")
        return []
    
    def _save_processed_acts(self):
        """Save the list of processed acts to file."""
        try:
            with open(self.processed_acts_file, 'w') as f:
                json.dump(self.processed_acts, f, indent=2)
        except Exception as e:
            print(f"Error saving processed acts: {e}")
    
    def _get_act_hash(self, act):
        """Generate a hash for an act to use as filename."""
        return hashlib.md5(act.encode()).hexdigest()
    
    def _load_act_documents(self, act):
        """Load documents for a specific act."""
        documents = []
        
        # Parse the act path
        act_parts = act.split('/')
        directory_path = os.path.join(self.legislation_folder_path, *act_parts)
        
        if not os.path.exists(directory_path):
            print(f"  Warning: Directory does not exist: {directory_path}")
            return documents
        
        # Load all .txt files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt') and ('section' in filename.lower() or 'schedule' in filename.lower()):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # Split large content into chunks
                    if len(content) > 50000:
                        chunks = self.text_splitter.split_text(content)
                        for i, chunk in enumerate(chunks):
                            doc = Document(page_content=chunk)
                            doc.metadata['legislation_id'] = act
                            doc.metadata['section_id'] = f"{filename.replace('.txt', '')}_chunk_{i}"
                            doc.metadata['original_section'] = filename.replace('.txt', '')
                            documents.append(doc)
                    else:
                        doc = Document(page_content=content)
                        doc.metadata['legislation_id'] = act
                        doc.metadata['section_id'] = filename.replace('.txt', '')
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
        
        return documents
    
    def process_act(self, act):
        """Process a single act and save its embeddings."""
        act_hash = self._get_act_hash(act)
        act_cache_dir = os.path.join(self.embeddings_cache_dir, act_hash)
        
        # Check if already processed
        if act in self.processed_acts and os.path.exists(act_cache_dir):
            print(f"  Act '{act}' already processed, skipping")
            return True
        
        print(f"  Processing act: {act}")
        
        # Load documents for this act
        documents = self._load_act_documents(act)
        
        if not documents:
            print(f"  Warning: No documents found for act '{act}'")
            return False
        
        # Create embeddings and save
        try:
            os.makedirs(act_cache_dir, exist_ok=True)
            
            # Build vector store for this act
            vector_store = openAIHandler.build_vector_db_with_batching(documents)
            
            # Save the vector store
            vector_store.save_local(act_cache_dir)
            
            # Mark as processed
            if act not in self.processed_acts:
                self.processed_acts.append(act)
                self._save_processed_acts()
            
            print(f"  Successfully processed act '{act}' with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"  Error processing act '{act}': {e}")
            return False
    
    def process_acts_batch(self, acts):
        """Process a batch of acts."""
        print(f"Processing batch of {len(acts)} acts...")
        
        successful = 0
        for act in tqdm(acts, desc="Processing acts"):
            if self.process_act(act):
                successful += 1
        
        print(f"Successfully processed {successful}/{len(acts)} acts")
        return successful
    
    def build_case_vector_store(self, case_name, case_acts):
        """Build a vector store for a specific case using only its relevant acts, by merging existing embeddings (no recomputation)."""
        print(f"Building vector store for case '{case_name}' with {len(case_acts)} acts (no recomputation)")
        
        # Ensure all acts are processed
        missing_acts = [act for act in case_acts if act not in self.processed_acts]
        if missing_acts:
            print(f"Processing {len(missing_acts)} missing acts: {missing_acts}")
            self.process_acts_batch(missing_acts)
        
        # Prepare to merge all act vector stores
        all_vectors = []
        all_docs = []
        all_metadatas = []
        docstore_class = None
        dim = None
        
        for act in case_acts:
            act_hash = self._get_act_hash(act)
            act_cache_dir = os.path.join(self.embeddings_cache_dir, act_hash)
            
            if os.path.exists(act_cache_dir):
                try:
                    act_vector_store = FAISS.load_local(act_cache_dir, self.embeddings_model, allow_dangerous_deserialization=True)
                    if dim is None:
                        dim = act_vector_store.index.d
                    if docstore_class is None:
                        docstore_class = act_vector_store.docstore.__class__
                    # For each vector in the act's index, reconstruct and collect
                    for vector_index, doc_id in act_vector_store.index_to_docstore_id.items():
                        doc = act_vector_store.docstore._dict[doc_id]
                        vector = act_vector_store.index.reconstruct(vector_index)
                        all_vectors.append(vector)
                        all_docs.append(doc)
                        all_metadatas.append(doc.metadata)
                except Exception as e:
                    print(f"Error loading act '{act}': {e}")
            else:
                print(f"Warning: Act '{act}' not found in cache")
        
        if not all_vectors:
            print(f"Warning: No vectors found for case '{case_name}'")
            return None

        def is_section_or_schedule(doc):
            section_id = doc.metadata.get('section_id', '').lower()
            original_section = doc.metadata.get('original_section', '').lower()
            return 'section' in section_id or 'schedule' in section_id or 'section' in original_section or 'schedule' in original_section

        # Filter all_docs and all_vectors together
        filtered = [(v, d) for v, d in zip(all_vectors, all_docs) if is_section_or_schedule(d)]
        if not filtered:
            print(f"  Warning: No documents found with section/schedule for case '{case_name}'")
            return None

        filtered_vectors, filtered_docs = zip(*filtered)

        # Create a new FAISS index and docstore
        new_index = faiss.IndexFlatL2(dim)
        new_docstore = docstore_class()
        new_index_to_docstore_id = {}

        for i, (vector, doc) in enumerate(zip(filtered_vectors, filtered_docs)):
            new_index.add(vector.reshape(1, -1))
            new_docstore._dict[str(i)] = doc
            new_index_to_docstore_id[i] = str(i)

        # Create the new FAISS store
        case_vector_store = FAISS(
            index=new_index,
            embedding_function=self.embeddings_model,
            docstore=new_docstore,
            index_to_docstore_id=new_index_to_docstore_id,
        )
        
        # Save the case-specific vector store
        case_store_dir = os.path.join(self.vector_stores_dir, case_name)
        os.makedirs(case_store_dir, exist_ok=True)
        case_vector_store.save_local(case_store_dir)
        
        print(f"Successfully built vector store for case '{case_name}' with {len(filtered_vectors)} vectors (no recomputation)")
        return case_vector_store
    
    def get_case_vector_store(self, case_name, case_acts):
        """
        Get a case-specific vector store.
        Tries to load from cache first, otherwise builds it by merging act embeddings.
        """
        case_cache_dir = os.path.join(self.vector_stores_dir, case_name)
        
        if os.path.exists(case_cache_dir):
            print(f"Loading existing vector store for case '{case_name}'")
            try:
                # The filtering is now done when acts are first processed.
                # We can directly load and return the cached store.
                vector_store = FAISS.load_local(case_cache_dir, self.embeddings_model, allow_dangerous_deserialization=True)
                print(f"  Successfully loaded cached store with {len(vector_store.docstore._dict)} documents.")
                return vector_store
            except Exception as e:
                print(f"Error loading case vector store: {e}")
                # If loading fails, proceed to build it
        
        # If not loaded, build it
        return self.build_case_vector_store(case_name, case_acts)
    
    def get_stats(self):
        """Get statistics about the managed acts and embeddings."""
        cache_dirs = [d for d in os.listdir(self.embeddings_cache_dir) 
                     if os.path.isdir(os.path.join(self.embeddings_cache_dir, d)) and d != '__pycache__']
        
        return {
            'total_processed_acts': len(self.processed_acts),
            'processed_acts': self.processed_acts,
            'cache_size': len(cache_dirs)
        }
    
    def clear_cache(self):
        """Clear all cached embeddings and vector stores."""
        
        
        if os.path.exists(self.embeddings_cache_dir):
            shutil.rmtree(self.embeddings_cache_dir)
            os.makedirs(self.embeddings_cache_dir, exist_ok=True)
        
        if os.path.exists(self.vector_stores_dir):
            shutil.rmtree(self.vector_stores_dir)
            os.makedirs(self.vector_stores_dir, exist_ok=True)
        
        self.processed_acts = []
        self._save_processed_acts()
        
        print("Cache cleared successfully")


# Example usage function
def example_usage():
    """Example of how to use the ActEmbeddingsManager."""
    from langchain_community.embeddings import OpenAIEmbeddings
    
    # Configuration
    legislation_folder_path = 'data/final_test/case_csvs/legislation'
    embeddings_cache_dir = 'act_embeddings_cache'
    vector_stores_dir = 'case_vector_stores'
    
    # Initialize
    embeddings_model = OpenAIEmbeddings()
    manager = ActEmbeddingsManager(
        legislation_folder_path=legislation_folder_path,
        embeddings_cache_dir=embeddings_cache_dir,
        vector_stores_dir=vector_stores_dir,
        embeddings_model=embeddings_model
    )
    
    # Example acts to process
    sample_acts = ['id/ukpga/2014/6', 'id/ukpga/2005/5']
    
    # Process acts
    manager.process_acts_batch(sample_acts)
    
    # Build vector store for a case
    case_name = 'example_case'
    case_acts = ['id/ukpga/2014/6','id/ukpga/2005/5']
    vector_store = manager.get_case_vector_store(case_name, case_acts)
    
    # Search for relevant sections
    if vector_store:
        results = vector_store.similarity_search_with_score(
            query="This is a sample query",
            k=2,
            filter={"legislation_id": {"$in": case_acts}}
        )
        print(f"Found {len(results)} relevant sections")
    
    # Show stats
    stats = manager.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    example_usage() 