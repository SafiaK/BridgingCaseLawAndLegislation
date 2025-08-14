import os
import pickle
import hashlib
import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import openAIHandler

import pandas as pd

import ast
from tqdm import tqdm

import faiss
from urllib.parse import urlparse

# Load environment variables from src/.env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.getcwd()), 'src', '.env'))

# Helper function to get base URL (copied from util.py to avoid import issues)
def get_base_url(uri):
    return urlparse(uri)._replace(path="", params="", query="", fragment="").geturl()


def load_legislative_sections(legislation_list, legislation_folder_path):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
    )

    for act_id in legislation_list:
        directory_path = os.path.join(legislation_folder_path, act_id)
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt') and ('section' in filename.lower() or 'schedule' in filename.lower()):
                    file_path = os.path.join(directory_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                        
                        chunks = text_splitter.split_text(content)
                        section_base_id = filename.replace('.txt', '')

                        for i, chunk in enumerate(chunks):
                            doc = Document(page_content=chunk)
                            doc.metadata['legislation_id'] = act_id
                            doc.metadata['section_id'] = f"{section_base_id}_chunk_{i}"
                            doc.metadata['original_section'] = section_base_id
                            all_docs.append(doc)
                            
                    except Exception as e:
                        print(f"Error reading or processing file {file_path}: {e}")
    
    return all_docs


# Helper to build vector store using openAIHandler's batched function
def build_vector_store(documents, embeddings_model,path):
    if not documents:
        print("Warning: No documents to build vector store")
        return None
    try:
        # Check if the vector store exists at the given path
        if os.path.exists(path):
            print(f"Loading existing FAISS vector store from: {path}")
            vector_store = load_faiss_vector_store(embeddings_model, path)
        else:
            print(f"No FAISS vector store found at: {path}. Building a new one.")
            vector_store = openAIHandler.build_vector_db_with_batching(documents)
            save_to_faiss(vector_store, path)
        return vector_store
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None

# Helper to get all sections mentioned explicitly in references
def get_all_the_sections_mentioned_explicitly(references, legislation_folder_path):
    all_docs = []
    if not references or not isinstance(references, list):
        return all_docs

    print(f"Trying to get all the explicitly mentioned sections from {len(references)} references.")

    for ref in references:
        # Handle case where ref might be a string instead of dict
        if isinstance(ref, str):
            url = ref
        elif isinstance(ref, dict) and 'href' in ref:
            url = ref['href']
        else:
            print(f"Warning: Skipping invalid reference format: {ref}")
            continue
            
        keywords = ['section', 'schedule']
        for keyword in keywords:
            if keyword in url.lower():
                if 'https' in url:
                    legislation_parts = url.split('https://www.legislation.gov.uk/', 1)[-1].split('/')
                else:
                    legislation_parts = url.split('http://www.legislation.gov.uk/', 1)[-1].split('/')
                act_parts = []
                section_parts = []
                found_keyword = False
                for part in legislation_parts:
                    if keyword in part.lower():
                        found_keyword = True
                        section_id = legislation_parts[-1]
                        if section_id.startswith('/'):
                            section_id = section_id[1:]
                        section_parts.append(part)
                    elif found_keyword:
                        section_parts.append(part)
                    else:
                        act_parts.append(part)
                if not found_keyword:
                    continue
                directory_path = os.path.join(legislation_folder_path, *act_parts)
                section_file_path = '-'.join(section_parts)
                file_name = f"{section_file_path}.txt"
                file_path = os.path.join(directory_path, file_name)
                try:
                    with open(file_path, 'r') as file:
                        content = file.read()
                    doc = Document(content)
                    doc.metadata['id'] = f"{'_'.join(act_parts)}_{keyword}_{section_id}"
                    doc.metadata['legislation_id'] = f"{'_'.join(act_parts)}_{keyword}_{section_id}"
                    all_sections.append(doc)
                    break
                except FileNotFoundError:
                    print(f"Warning: Could not find file for {url} at {file_path}")
                    continue
    return all_sections

# Helper to get relevant sections from vector store
def get_relevantSections_from_vectore_store(query, legislation_filter_list, vectore_store):
    def is_section_or_schedule(doc):
        section_id = doc.metadata.get('section_id', '').lower()
        original_section = doc.metadata.get('original_section', '').lower()
        return 'section' in section_id or 'schedule' in section_id or 'section' in original_section or 'schedule' in original_section

    all_docs_with_scores = []
    print("The searching list is ",legislation_filter_list)
    print("---------------------------------------------")
    try:
        for legislation in legislation_filter_list:
            results = vectore_store.similarity_search_with_score(
                query=query,
                k=2,
                filter={"legislation_id": legislation}
            )
            print("Results from",results)
            if len(results) > 0:
                all_docs_with_scores.extend(results)
        top_section_schedule = [(doc, score) for doc, score in all_docs_with_scores if is_section_or_schedule(doc)][:2]

        if top_section_schedule:
            # Use these as your matches
            for best_doc, best_score in top_section_schedule:
                print("Selected:", best_doc.metadata.get('section_id'), best_score)
        else:
            # Handle as no match
            print("No section/schedule")
        top_section_schedule.sort(key=lambda x: x[1])
        top_docs = top_section_schedule[:2]
        relevant_docs = [doc for doc, score in top_docs]
        print("The all_docs_with_scores for is",top_section_schedule)
        print("---------------------------------------------")

    except Exception as e:
        print(f"Error in get_relevantSection: {e}")
        relevant_docs = []
    return relevant_docs

# Main function to use in your loop - now includes vector store building
def get_the_relevant_sections(query, ref_list, references, legislation_folder_path,vectore_store):
    docs = []
    

    print("The query is",query)
    print("========================================")
    print("The ref_list is",ref_list)
    print("========================================")
    print("The references is",references)
    print("========================================")
    print("The legislation_folder_path is",legislation_folder_path)
    print("========================================")
    
    # Safely parse references if it's a string
    parsed_references = []
    if isinstance(references, str) and references.strip():
        try:
            # It might be a JSON string representation of a list
            parsed_references = json.loads(references)
        except json.JSONDecodeError:
            try:
                # Or a Python literal string
                parsed_references = ast.literal_eval(references)
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse references string: {references[:100]}")
    elif isinstance(references, list):
        parsed_references = references
    
    try:
        docs_through_explicit_mention = get_all_the_sections_mentioned_explicitly(parsed_references, legislation_folder_path)
        if docs_through_explicit_mention:
            docs.extend(docs_through_explicit_mention)
        
        # Only use vector store if it was successfully built
        if vectore_store:
            # Handle references properly - check if it's a list of dicts
            base_urls = []
            if isinstance(parsed_references, list):
                base_urls = [get_base_url(ref['href']) for ref in parsed_references if isinstance(ref, dict) and 'href' in ref]
                
            candidate_url_list = list(set(ref_list) - set(base_urls))
            print("The candidate url list is ",candidate_url_list)

            if candidate_url_list:
                docs_through_vector_db = get_relevantSections_from_vectore_store(query, candidate_url_list, vectore_store)
                docs.extend(docs_through_vector_db)
        else:
            raise RuntimeError("vectore_store is not available and cannot be built in this context.")
    except Exception as e:
        print(f"Error in main logic: {e}")
        # Fallback: try vector store search with all legislation
        if vectore_store:
            docs_through_vector_db = get_relevantSections_from_vectore_store(query, ref_list, vectore_store)
            docs.extend(docs_through_vector_db)
    
    return docs


def save_to_faiss(faiss_vector_database, db_path="faiss_db"):
    faiss_vector_database.save_local(db_path)

def load_faiss_vector_store(embeddings,db_directory_path="faiss_db"):
    embeddings = embeddings
    vector_store = FAISS.load_local(db_directory_path, embeddings, allow_dangerous_deserialization=True)
    #retriever = vector_store.as_retriever()
    # Filter existing vector store to keep only documents with section/schedule in metadata
    all_docs = vector_store.similarity_search("", k=100000)  # Get all documents
    
    filtered_docs = []
    for doc in all_docs:
        section_id = doc.metadata.get('section_id', '').lower()
        original_section = doc.metadata.get('original_section', '').lower()
        
        # Check if section_id or original_section contains "section" or "schedule"
        if ('section' in section_id or 'schedule' in section_id or 
            'section' in original_section or 'schedule' in original_section):
            filtered_docs.append(doc)
    
    print(f"Filtered from {len(all_docs)} to {len(filtered_docs)} documents with section/schedule in metadata")
    
    if filtered_docs:
        # Create new vector store with filtered documents
        filtered_vector_store = FAISS.from_documents(filtered_docs, embeddings)
        return filtered_vector_store
    else:
        print("Warning: No documents found with section/schedule in metadata")
        return vector_store

    return vector_store
def _create_case_clusters(case_legislation_map, max_acts_per_cluster=50):
    """
    Groups cases into clusters based on shared legislation, with a max number of acts per cluster.
    """
    clusters = []
    # Create a copy of the cases to process, as we will be removing items
    remaining_cases = list(case_legislation_map.keys())

    while remaining_cases:
        # Start a new cluster seeded with the first remaining case
        current_cluster_cases = []
        seed_case = remaining_cases.pop(0)
        current_cluster_cases.append(seed_case)
        
        # Initialize the set of acts for the current cluster
        cluster_acts = set(case_legislation_map[seed_case])

        # Greedily add more cases to the cluster
        # Iterate through a copy of remaining_cases to allow safe removal
        for case in list(remaining_cases):
            case_acts = set(case_legislation_map.get(case, []))
            
            # Check if adding this case would exceed the act limit
            if len(cluster_acts.union(case_acts)) <= max_acts_per_cluster:
                # If not, add the case to the cluster and its acts to the set
                current_cluster_cases.append(case)
                cluster_acts.update(case_acts)
                remaining_cases.remove(case)
        
        clusters.append(current_cluster_cases)
        
    print(f"Created {len(clusters)} case clusters.")
    return clusters

def get_the_acts_of_this_cluster():
    pass

def find_sections_for_dataframe_with_clustering(df, case_legislation_map=None, legislation_folder_path=None, embeddings_model=None, combined_map_path='../data/final_test/case_csvs/combined_case_legislation_map.pkl'):
    """
    The main function to process a dataframe by clustering cases to manage resources.
    
    Args:
        df (pd.DataFrame): The input DataFrame with case and paragraph data.
        case_legislation_map (dict, optional): A dictionary mapping case_name to a list of legislation IDs.
        legislation_folder_path (str): The path to the root legislation folder.
        embeddings_model: The embeddings model to use for vector store creation.
        combined_map_path (str): Path to the combined case legislation map pickle file.
    """
    # Load the combined case legislation map if not provided
    if case_legislation_map is None:
        case_legislation_map = load_combined_case_legislation_map(combined_map_path)
    
    if not case_legislation_map:
        print("Error: No case legislation map available")
        df['section_id'] = None
        df['section_text'] = None
        return df
    
    # Step 1: Create clusters of cases
    filtered_case_legislation_map = {k: v for k, v in case_legislation_map.items() if k in df['case_name'].unique()}
    case_clusters = _create_case_clusters(filtered_case_legislation_map)
    
    all_results = []

    # Step 2: Process each cluster
    for i, cluster_cases in enumerate(case_clusters):
        print(f"\n--- Processing Cluster {i+1}/{len(case_clusters)} ({len(cluster_cases)} cases) ---")
        
        # Get the subset of the DataFrame for the current cluster
        cluster_df = df[df['case_name'].isin(cluster_cases)].copy()
        
        if cluster_df.empty:
            print("No paragraphs found for this cluster. Skipping.")
            continue

        # Compute the union of all acts for this cluster
        cluster_acts = set()
        for case in cluster_cases:
            cluster_acts.update(case_legislation_map.get(case, []))
        # Save the acts for this cluster (optional, for reusability)
        with open(f'vector_cache/cluster_{i+1}_acts.txt', 'w') as f:
            for act in sorted(cluster_acts):
                f.write(f'{act}\n')
        # Build a cluster-specific map
        cluster_case_legislation_map = {case: case_legislation_map[case] for case in cluster_cases}

        try:
            # Process the smaller DataFrame, passing cluster_acts
            updated_cluster_df = process_dataframe_for_sections(
                cluster_df,
                cluster_case_legislation_map,
                legislation_folder_path,
                embeddings_model,
                cluster_acts=list(cluster_acts)
            )
            # Append the results (only the necessary columns)
            all_results.append(updated_cluster_df[['para_id', 'section_id', 'section_text']])
        except Exception as e:
            print(f"Error processing cluster {i+1}: {e}")
            if all_results:
                print("Returning all successfully processed rows so far.")
                final_results_df = pd.concat(all_results, ignore_index=True)
                merged_df = pd.merge(df, final_results_df, on='para_id', how='left')
                return merged_df
            else:
                print("No clusters processed successfully. Returning original DataFrame.")
                df['section_id'] = None
                df['section_text'] = None
                return df

    # Step 3: Combine all results and merge with the original DataFrame
    if not all_results:
        print("No results were generated from any cluster.")
        df['section_id'] = None
        df['section_text'] = None
        return df

    final_results_df = pd.concat(all_results, ignore_index=True)
    merged_df = pd.merge(df, final_results_df, on='para_id', how='left')
    return merged_df
import pandas as pd
from tqdm import tqdm
from ActEmbeddingsManager import ActEmbeddingsManager


def process_dataframe_case_by_case_with_act_manager(
    df,
    legislation_folder_path,
    embeddings_cache_dir,
    vector_stores_dir,
    embeddings_model,
    combined_map_path='../data/final_test/case_csvs/combined_case_legislation_map.pkl'
):
    """
    Processes a DataFrame case by case, using ActEmbeddingsManager to build/load per-case vector stores.
    Returns a DataFrame with section_id and section_text columns.
    """
    # Load the combined case legislation map
    case_legislation_map = load_combined_case_legislation_map(combined_map_path)

    manager = ActEmbeddingsManager(
        legislation_folder_path=legislation_folder_path,
        embeddings_cache_dir=embeddings_cache_dir,
        vector_stores_dir=vector_stores_dir,
        embeddings_model=embeddings_model
    )

    results = []
    for case_name, case_df in tqdm(df.groupby('case_name'), desc="Processing cases"):
        case_acts = case_legislation_map.get(case_name, [])
        if not case_acts:
            # No acts for this case, fill with None
            for _, row in case_df.iterrows():
                results.append({
                    'para_id': row.get('para_id'),
                    'section_id': None,
                    'section_text': None
                })
            continue

        vector_store = manager.get_case_vector_store(case_name, case_acts)
        if vector_store is None:
            print("The vector store is None")
            print("===========================")
            for _, row in case_df.iterrows():
                results.append({
                    'para_id': row.get('para_id'),
                    'section_id': None,
                    'section_text': None
                })
            continue

        for _, row in case_df.iterrows():
            para_text = row.get('paragraphs', '')
            relevant_sections = get_the_relevant_sections(
                query=para_text,
                ref_list=case_acts,
                references=row.get('references', []),
                legislation_folder_path=legislation_folder_path,
                vectore_store=vector_store
            )
            
            if relevant_sections:
                for best_doc in relevant_sections:
                    print("The search results are not None")
                    print("===========================")
                    section_id = f"{best_doc.metadata.get('legislation_id', '')}_{best_doc.metadata.get('section_id', '')}"
                    section_text = best_doc.page_content
            
                    results.append({
                        'para_id': row.get('para_id'),
                        'section_id': section_id,
                        'section_text': section_text
                    })
            else:
                results.append({
                    'para_id': row.get('para_id'),
                    'section_id': None,
                    'section_text': None
                })
        # Optionally: delete the vector store from disk to save space
        # import shutil; shutil.rmtree(manager.vector_stores_dir + '/' + case_name, ignore_errors=True)

    results_df = pd.DataFrame(results)
    merged_df = pd.merge(df, results_df, on='para_id', how='left')
    return merged_df
def process_dataframe_for_sections(df, case_legislation_map, legislation_folder_path, embeddings_model, cluster_acts=None, combined_map_path='../data/final_test/case_csvs/combined_case_legislation_map.pkl'):
    """
    Processes a DataFrame to find and append relevant legislation sections for each row.
    Args:
        df (pd.DataFrame): The input DataFrame with case and paragraph data.
        case_legislation_map (dict): A dictionary mapping case_name to a list of legislation IDs.
        legislation_folder_path (str): The path to the root legislation folder.
        embeddings_model: The embeddings model to use for vector store creation.
        cluster_acts (list or None): If provided, use this list of acts for all paragraphs in the cluster.
        combined_map_path (str): Path to the combined case legislation map pickle file.
    Returns:
        pd.DataFrame: The DataFrame with 'section_id' and 'section_text' columns appended.
    """
    # Load the combined case legislation map if not provided
    if case_legislation_map is None:
        case_legislation_map = load_combined_case_legislation_map(combined_map_path)
    
    if not case_legislation_map:
        print("Error: No case legislation map available")
        df['section_id'] = None
        df['section_text'] = None
        return df
    results = []
    if embeddings_model and cluster_acts:
        documents = load_legislative_sections(cluster_acts, legislation_folder_path)
        faiss_db_dir = os.path.join(legislation_folder_path, "faiss_db")
        #os.makedirs(faiss_db_dir, exist_ok=True)

        if os.path.exists(faiss_db_dir):
            vectore_store = load_faiss_vector_store( embeddings_model,faiss_db_dir)
            existing_ids = get_legislation_ids_in_store(vectore_store)
            missing_acts = set(cluster_acts) - existing_ids
            if missing_acts:
                print(f"Adding missing acts to vector store: {missing_acts}")
                documents = load_legislative_sections(list(missing_acts), legislation_folder_path)
                batch_size = 50
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    vectore_store.add_documents(batch)
                # Optionally, save the updated store
                save_to_faiss(vectore_store,faiss_db_dir)
        else:
            documents = load_legislative_sections(cluster_acts, legislation_folder_path)
            vectore_store = build_vector_store(documents, embeddings_model,faiss_db_dir)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing paragraphs"):
        case_name = row.get('case_name', '')
        para_text = row.get('paragraphs', '')
        references = row.get('references', [])

        # Always use the acts specific to this case, not cluster acts
        legislation_list = case_legislation_map.get(case_name, [])

        if not legislation_list:
            # Append empty results if no legislation is mapped to the case
            results.append({
                'para_id': row.get('para_id'),
                'section_id': None,
                'section_text': None
            })
            continue

        # Get the matching sections for this paragraph
        relevant_sections = get_the_relevant_sections(
            query=para_text,
            ref_list=legislation_list,
            references=references,
            legislation_folder_path=legislation_folder_path,
            vectore_store =vectore_store,
            embeddings_model=embeddings_model
        )

        if relevant_sections:
            # For simplicity, we take the first and most relevant section found
            best_section = relevant_sections[0]
            results.append({
                'para_id': row.get('para_id'),
                'section_id': f"{best_section.metadata.get('legislation_id', '')}_{best_section.metadata.get('section_id', '')}",
                'section_text': best_section.page_content
            })
        else:
            results.append({
                'para_id': row.get('para_id'),
                'section_id': None,
                'section_text': None
            })
            
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Merge the results back into the original DataFrame
    # This requires a unique identifier in the original df, like 'para_id'
    if 'para_id' in df.columns:
        # If there's a para_id, we can merge on it
        merged_df = pd.merge(df, results_df, on='para_id', how='left')
    else:
        # Otherwise, we assume the index aligns and join on it
        merged_df = df.join(results_df.set_index('para_id'))

    return merged_df 

def get_legislation_ids_in_store(vectore_store):
    return set(doc.metadata.get('legislation_id', '') for doc in vectore_store.docstore._dict.values()) 

def get_acts_for_case(case_name, case_legislation_map=None, combined_map_path='../data/final_test/case_csvs/combined_case_legislation_map.pkl'):
    """
    Get the list of legislation acts for a specific case name.
    
    Args:
        case_name (str): The name of the case (e.g., 'ewhc_fam_2018_3244')
        case_legislation_map (dict, optional): Pre-loaded case legislation map
        combined_map_path (str): Path to the combined case legislation map pickle file
        
    Returns:
        list: List of legislation acts for the case, or empty list if not found
    """
    if case_legislation_map is None:
        case_legislation_map = load_combined_case_legislation_map(combined_map_path)
    
    return case_legislation_map.get(case_name, [])

def find_sections_for_dataframe_with_case_specific_stores(df, case_legislation_map=None, legislation_folder_path=None, embeddings_model=None, combined_map_path='../data/final_test/case_csvs/combined_case_legislation_map.pkl'):
    """
    Process a DataFrame using case-specific vector stores built from the combined case legislation map.
    Each case gets its own vector store containing only the acts relevant to that case.
    
    Args:
        df (pd.DataFrame): The input DataFrame with case and paragraph data.
        case_legislation_map (dict, optional): A dictionary mapping case_name to a list of legislation IDs.
        legislation_folder_path (str): The path to the root legislation folder.
        embeddings_model: The embeddings model to use for vector store creation.
        combined_map_path (str): Path to the combined case legislation map pickle file.
        
    Returns:
        pd.DataFrame: The DataFrame with 'section_id' and 'section_text' columns appended.
    """
    # Load the combined case legislation map if not provided
    def is_section_or_schedule(doc):
        section_id = doc.metadata.get('section_id', '').lower()
        original_section = doc.metadata.get('original_section', '').lower()
        return 'section' in section_id or 'schedule' in section_id or 'section' in original_section or 'schedule' in original_section

    if case_legislation_map is None:
        case_legislation_map = load_combined_case_legislation_map(combined_map_path)
    
    if not case_legislation_map:
        print("Error: No case legislation map available")
        df['section_id'] = None
        df['section_text'] = None
        return df
    
    # Import ActEmbeddingsManager here to avoid circular imports
    try:
        from ActEmbeddingsManager import ActEmbeddingsManager
    except ImportError:
        print("Error: ActEmbeddingsManager not found. Please run the ActEmbeddingsManager.ipynb notebook first.")
        df['section_id'] = None
        df['section_text'] = None
        return df
    
    # Initialize the manager
    embeddings_cache_dir = os.path.join(legislation_folder_path, "act_embeddings_cache")
    vector_stores_dir = os.path.join(legislation_folder_path, "case_vector_stores")
    
    manager = ActEmbeddingsManager(
        legislation_folder_path=legislation_folder_path,
        embeddings_cache_dir=embeddings_cache_dir,
        vector_stores_dir=vector_stores_dir,
        embeddings_model=embeddings_model
    )
    
    results = []
    
    # Group by case to process efficiently
    case_groups = df.groupby('case_name')
    
    for case_name, case_df in tqdm(case_groups, total=len(case_groups), desc="Processing cases"):
        print(f"\nProcessing case: {case_name}")
        
        # Get acts for this case from the combined map
        case_acts = case_legislation_map.get(case_name, [])
        
        if not case_acts:
            print(f"  No acts found for case '{case_name}', skipping")
            # Add empty results for this case
            for _, row in case_df.iterrows():
                results.append({
                    'para_id': row.get('para_id'),
                    'section_id': None,
                    'section_text': None
                })
            continue
        
        # Get or build vector store for this case
        vector_store = manager.get_case_vector_store(case_name, case_acts)
        
        if vector_store is None:
            print(f"  Could not build vector store for case '{case_name}'")
            # Add empty results for this case
            for _, row in case_df.iterrows():
                results.append({
                    'para_id': row.get('para_id'),
                    'section_id': None,
                    'section_text': None
                })
            continue
        
        # Process each paragraph in this case
        for _, row in case_df.iterrows():
            para_text = row.get('paragraphs', '')
            references = row.get('references', [])
            
            # --- FIX: Truncate oversized query text to prevent token limit errors ---
            max_query_length = 30000 
            if len(para_text) > max_query_length:
                print(f"  Warning: Truncating oversized paragraph (para_id: {row.get('para_id')}) from {len(para_text)} to {max_query_length} chars.")
                para_text = para_text[:max_query_length]
            # --- END FIX ---

            # Search for relevant sections using only this case's acts
            try:
                search_results = vector_store.similarity_search_with_score(
                    query=para_text,
                    k=100,
                    filter={"legislation_id": {"$in": case_acts}}
                )
                # Pick the top 2 that are section/schedule
                top_section_schedule = [(doc, score) for doc, score in search_results if is_section_or_schedule(doc)][:2]

                if top_section_schedule:
                    # Use the best of these as your match
                    best_doc, best_score = top_section_schedule[0]
                    results.append({
                        'para_id': row.get('para_id'),
                        'section_id': f"{best_doc.metadata.get('legislation_id', '')}_{best_doc.metadata.get('section_id', '')}",
                        'section_text': best_doc.page_content
                    })
                else:
                    results.append({
                        'para_id': row.get('para_id'),
                        'section_id': None,
                        'section_text': None
                    })
                    print("No section/schedule found in top 100")
            except Exception as e:
                print(f"  Error processing paragraph: {e}")
                results.append({
                    'para_id': row.get('para_id'),
                    'section_id': None,
                    'section_text': None
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original DataFrame
    if 'para_id' in df.columns:
        merged_df = pd.merge(df, results_df, on='para_id', how='left')
    else:
        merged_df = df.join(results_df.set_index('para_id'))
    
    return merged_df

def load_combined_case_legislation_map(map_path='../data/final_test/case_csvs/combined_case_legislation_map.pkl'):
    """
    Load the combined case legislation map from pickle file.
    
    Args:
        map_path (str): Path to the combined case legislation map pickle file
        
    Returns:
        dict: Dictionary mapping case names to lists of legislation acts
    """
    try:
        with open(map_path, 'rb') as f:
            case_legislation_map = pickle.load(f)
        print(f"Loaded combined case legislation map with {len(case_legislation_map)} cases")
        return case_legislation_map
    except FileNotFoundError:
        print(f"Warning: Combined case legislation map not found at {map_path}")
        return {}
    except Exception as e:
        print(f"Error loading combined case legislation map: {e}")
        return {} 