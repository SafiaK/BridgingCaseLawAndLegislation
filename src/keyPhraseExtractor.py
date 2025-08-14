import pandas as pd
import openAIHandler
import pickle
import util
import json 
import re
import time
import os
import ast
from langchain.docstore.document import Document
import csv

# Initialize the FAISS vector store
global vectore_store


def get_the_relevant_sections(query, ref_list,references,legislation_folder_path):
    #get all the sections/schdule/chapters/ mentioned in the referece
    #get the matching section as well from the vectore store
    docs = []
    try:
        docs_through_explicit_mention = get_all_the_sections_mentioned_explicitly(references,legislation_folder_path)
        if docs_through_explicit_mention:
                docs.extend(docs_through_explicit_mention)
                print("================================================================")
                print(" Recieved explict references for paragraphs",query)
                print("================================================================")
               # print(docs_through_explicit_mention)

        #get the bas urls from references

        # Get base URLs from references
        base_urls = [util.get_base_url(ref['href']) for ref in references]
        #call get_relevantSections_from_vectore_store on only those urls in ref_list which are not in base_urls
        # if nothing such url then don't call get_relevantSections_from_vectore_store
        candidate_url_list = list(set(ref_list) - set(base_urls))
        if set(ref_list) - set(base_urls):
            docs_through_vector_db = get_relevantSections_from_vectore_store(query, list(set(ref_list) - set(base_urls)))
            docs.extend(docs_through_vector_db)
    except Exception as e:
        print(e)
        docs_through_vector_db = get_relevantSections_from_vectore_store(query,ref_list)
        docs.extend(docs_through_vector_db)

    return docs

def get_all_the_sections_mentioned_explicitly(ref_list, legislation_folder_path):
    """
    Extracts and retrieves section content from legislation references.
    
    Args:
        ref_list: List of reference dictionaries containing legislation URLs
        legislation_folder_path: Path to the folder containing legislation files
        
    Returns:
        List of Document objects containing section content and metadata
    """
    all_sections = []
    
    for ref in ref_list:
        url = ref['href']
        
        keywords = ['section', 'schedule', 'regulation', 'chapter', 'article']
        
        for keyword in keywords:
            if keyword in url.lower():
                # Extract legislation parts from URL
                if 'https' in url:
                    legislation_parts = url.split('https://www.legislation.gov.uk/', 1)[-1].split('/')
                else:
                    legislation_parts = url.split('http://www.legislation.gov.uk/', 1)[-1].split('/')
                
                # Extract the act part (everything before the keyword)
                act_parts = []
                section_parts = []
                found_keyword = False
                
                for part in legislation_parts:
                    if keyword in part.lower():
                        found_keyword = True
                        # Extract section number from the part containing the keyword
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
                
                # Create base directory path
                directory_path = os.path.join(legislation_folder_path, *act_parts)
                
                # Create full file path
                section_file_path = '-'.join(section_parts)
                file_name = f"{section_file_path}.txt"
                file_path = os.path.join(directory_path, file_name)
                
                try:
                    # Open and read the section file
                    with open(file_path, 'r') as file:
                        content = file.read()
                    
                    # Create document with metadata
                    doc = Document(content)
                    doc.metadata['id'] = f"{'_'.join(act_parts)}_{keyword}_{section_id}"
                    doc.metadata['legislation_id'] = url
                    all_sections.append(doc)
                    break
                except FileNotFoundError:
                    print(f"Warning: Could not find file for {url} at {file_path}")
                    continue
    
    return all_sections
def getTheFirstSection(ref_list,legislation_folder_path):
    for ref in ref_list:
        act,section = ref['legislation_section']
        if section:
            
            section_u = section.split('/')[0]
            #print(f"the section is {section_u}")
            
            with open(f'{legislation_folder_path}/{act}/section-{section_u}.txt', 'r') as file:
                content = file.read()
                doc = Document(content)
                doc.metadata['id'] = f'{act}_section_{section_u}'
                doc.metadata['legislation_id'] = '{act}'
                return doc
    return None
def get_relevantSections_from_vectore_store(query, legislation_filter_list):
    all_docs_with_scores = []
    global vectore_store
    try:
        for legislation in legislation_filter_list:
            results = vectore_store.similarity_search_with_score(
                query=query,
                k=2,
                filter={"legislation_id": legislation}
            )
            if len(results) > 0:
                print("================================================")
                print(results)
                all_docs_with_scores.extend(results)  # Collect all docs with their scores
                
        # Sort all documents by score (ascending - lower score is better)
        all_docs_with_scores.sort(key=lambda x: x[1])
        
        # Take only the top 3 documents with lowest scores
        top_docs = all_docs_with_scores[:3]
        
        # Extract just the documents without scores for return
        relevant_docs = [doc for doc, score in top_docs]
            
    except Exception as e:
        print(f"Error in get_relevantSection: {e}")
        relevant_docs = []  # Return empty list if error occurred in search
            
    return relevant_docs
def process_case_annotations(case_number, input_file_path, output_file_path, case_legislation_dic, legislation_dir):
    test_case = input_file_path
    print(f"processing {test_case}")
    annotations_df_gpt = pd.read_csv(test_case)
    annotations_df_gpt['section_id'] = '0'
    annotations_df_gpt['section_text'] = ''
    legislation_list = case_legislation_dic[case_number]
    #annotations_df_gpt['final_annotation'] = annotations_df_gpt['final_annotation'].astype(str)
    new_rows = []

    for i, row in annotations_df_gpt.iterrows():
        if row['final_annotation'] == True:
            try:
                paragraph = row['paragraphs']
                references = row.get('references', [])
                references = ast.literal_eval(references)

                if len(references) > 0:
                    # Extract legislation sections from references if available
                    for ref in references:
                        if isinstance(ref, dict) and 'legislation_section' in ref:
                            legislation_id, section = ref['legislation_section']
                            if legislation_id:
                                relevant_docs = get_the_relevant_sections(paragraph, [legislation_id], references, legislation_dir)
                                for relevant_doc in relevant_docs:
                                    new_row = row.copy()
                                    section_id = relevant_doc.metadata.get("id", "unknown")
                                    section_text = str(relevant_doc.page_content)
                                    new_row['section_id'] = section_id
                                    new_row['section_text'] = section_text
                                    new_rows.append(new_row)
                                break
                else:
                    # Fall back to original behavior if no references
                    relevant_docs = get_the_relevant_sections(paragraph, legislation_list, references, legislation_dir)
                    for relevant_doc in relevant_docs:
                        new_row = row.copy()
                        section_id = relevant_doc.metadata.get("id", "unknown")
                        section_text = str(relevant_doc.page_content)
                        new_row['section_id'] = section_id
                        new_row['section_text'] = section_text
                        new_rows.append(new_row)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                pass

    if new_rows:
        new_annotations_df_gpt = pd.DataFrame(new_rows)
    else:
        new_annotations_df_gpt = annotations_df_gpt

    # Pre-clean section_text to remove/replace newlines and quotes
    def clean_text(text):
        if isinstance(text, str):
            try:
                text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                text = text.replace('"', "'")
                text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                text = text.replace('"', "'")
                
        
                
                text = text.replace(r'\s+', ' ')
            except Exception as e:
                print(e)
        return text
    if 'section_text' in new_annotations_df_gpt.columns:
        new_annotations_df_gpt['section_text'] = new_annotations_df_gpt['section_text'].apply(clean_text)

    output_file = output_file_path
    new_annotations_df_gpt.to_csv(output_file_path,
                                index=False,
                                quoting=csv.QUOTE_ALL,
                                escapechar='\\',
                                doublequote=True,
                                encoding='utf-8')
def getJsonList(results_str):
    try:
        results = json.loads(results_str)
        return results
    except:
        match = re.search(r'```json\n(.*?)\n```', results_str, re.S)
        if match:
            json_string = match.group(1)
            try:
                # Parse the extracted JSON string
                json_data = json.loads(json_string)
                print("Successfully extracted JSON list:")
                return json_data

           
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
                return []
def processToGetTriples(llm_chain_extraction,input_file_path, output_file_path):
    print(f"processing to extract phrases from file {input_file_path}")
    annotations_df_gpt = pd.read_csv(input_file_path,index_col=False)
    annotations_df_gpt['triples_result'] = ''
    for i ,row in annotations_df_gpt.iterrows():
        para_id =row['para_id']
        case_text = row['paragraphs']
        legislation_text = row['section_text']
        section_id = row['section_id']
        
        if section_id != '0':
            try:
                RESULTS = openAIHandler.getInterPretations(legislation_text,case_text,llm_chain_extraction)
                print(para_id)
                print(section_id)
                print("===========================")

                results = getJsonList(RESULTS)
                #RESULTS_legit = getIflegit(results,case_text,legislation_text)
                #print(RESULTS_legit)
                annotations_df_gpt.at[i, 'triples_result'] = results
            except Exception as e:
                print(f"Error occurred: {e}")
                continue
    #annotations_df_gpt.to_csv(output_file_path,index=False)
    # In processToGetTriples() function, replace the final csv writing line with:

    annotations_df_gpt.to_csv(output_file_path, 
                            index=False,
                            quoting=csv.QUOTE_NONNUMERIC,
                            escapechar='\\',
                            doublequote=True,
                            encoding='utf-8')

def getTheInterpretationDf(dataframe):
    # Filter rows where 'triples_result' is not NaN
    filtered_df = dataframe[dataframe['triples_result'].notna()]

    # Initialize a list to store the extracted data
    extracted_data = []

    # Iterate over each row in the filtered DataFrame
    for _, row in filtered_df.iterrows():
        # Parse the 'triples_result' JSON string into a list of dictionaries
        triples = ast.literal_eval(row['triples_result'])

        # Extract relevant fields from each triple
        for triple in triples:
            try:
                legislation_phrases =triple['key_phrases/concepts']
            except:
                legislation_phrases = triple['key_phrases']
                

            case_term = triple.get('case_law_term', '')
            legislation_term = triple.get('legislation_term', '')
            confidence = triple.get('confidence', '')
            reasoning = triple.get('reasoning', '')
            #legislation_phrases = triple.get('key_phrases/concepts', [])
            

            # Append the extracted data along with additional information to the list
            extracted_data.append({
                'url': row.get('case_uri', ''),
                'para_id': row.get('para_id', ''),
                'paragraphs': row.get('paragraphs', ''),
                'case_term_phrases': row.get('application_of_law_phrases', ''),
                'legislation_id': row.get('section_id', ''),
                'section_text':row.get('section_text', ''),
                'case_term': case_term,
                'legislation_term': legislation_term,
                'confidence': confidence,
                'reasoning': reasoning,
                'key_phrases': legislation_phrases
            })

    # Create a new DataFrame from the extracted data
    new_dataframe = pd.DataFrame(extracted_data)

    # Return the new DataFrame
    return new_dataframe
def getTheLegitPhrases(case_input_file_path,case_output_file_path):
    def checkIfPhraseInText(phrase,text):
        try:
            if phrase in text:
                return True
            else:
                return False
        except:
            return False
    print("=====================================")
    print(f"processing to extract phrases from file {case_input_file_path}")
    print("=====================================")
    data = pd.read_csv(case_input_file_path,index_col=False)
    data = getTheInterpretationDf(data)

    data_expanded = data.explode('key_phrases')
    data_expanded = data_expanded.dropna(subset='key_phrases')
    data_expanded['section_text'] = data_expanded['section_text'].astype(str)
    data_expanded['key_phrases'] = data_expanded['key_phrases'].astype(str)
    data_expanded['in_section_text'] = data_expanded.apply(
        lambda row: checkIfPhraseInText(row['key_phrases'], row['section_text']),axis=1)
    data_expanded = data_expanded[data_expanded['in_section_text']==True]
    data_expanded.drop(columns=['in_section_text'], inplace=True)
    data_expanded.to_csv(case_output_file_path,index=False)

def get_the_final_files(input_folder, output_folder):
    """
    Process CSV files in the input folder and separate rows based on specific conditions.

    Args:
        input_folder (str): Path to the folder containing input CSV files.
        output_folder (str): Path to the folder where output CSV files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    # Initialize lists to store rows
    rows_with_phrases = []
    rows_without_phrases = []

    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure "if_law_applied" column is treated as string
        df["final_annotation"] = df["final_annotation"].astype(str)

        # Filter rows where "if_law_applied" equals '1'
        filtered_rows = df[df["final_annotation"] == True]

        # Separate rows based on "triples_result" column
        for _, row in filtered_rows.iterrows():
            # Convert "triples_result" to a Python object if it's a string representation of a list
            try:
                triples_result = ast.literal_eval(row["triples_result"])
            except (ValueError, SyntaxError):
                triples_result = None

            # Check if "triples_result" is a non-empty list
            if isinstance(triples_result, list) and len(triples_result) > 0:
                rows_with_phrases.append(row)
            else:
                rows_without_phrases.append(row)

    # Convert lists to DataFrames
    df_with_phrases = pd.DataFrame(rows_with_phrases)
    df_without_phrases = pd.DataFrame(rows_without_phrases)

    # Save the DataFrames to output folder
    with_phrases_path = os.path.join(output_folder, 'rows_with_phrases.csv')
    without_phrases_path = os.path.join(output_folder, 'rows_without_keyPhrases.csv')

    df_with_phrases.to_csv(with_phrases_path, index=False)
    df_without_phrases.to_csv(without_phrases_path, index=False)

    print(f"Saved rows with phrases to: {with_phrases_path}")
    print(f"Saved rows without key phrases to: {without_phrases_path}")

    return with_phrases_path,without_phrases_path


def extractThePhrases(case_act_pickle_file,input_dir,output_dir,legislation_dir,output_folder_path_for_aggregated_result):
    print("Key Phrase Extractor is running...")
    with open(case_act_pickle_file, 'rb') as f:
        case_legislation_dic = pickle.load(f)
    acts = list(set(util.flatten_list_of_lists(case_legislation_dic.values())))
    global vectore_store
    vectore_store = openAIHandler.BuildVectorDB(legislation_dir,acts)
    case_list = list(case_legislation_dic.keys()) 
    sections_dir = f"{output_dir}/csv_with_legislation"
    os.makedirs(sections_dir, exist_ok=True)
    llm_chain_extraction = openAIHandler.getPhraseExtractionChain()
    
    for case_number in case_list:
        interpreted_file = f"{input_dir}/{case_number}.csv"
        interpreted_file_with_sections = f"{sections_dir}/{case_number}.csv"
        process_case_annotations(case_number, interpreted_file, interpreted_file_with_sections, case_legislation_dic,legislation_dir)
    
   
    # for case_number in case_list:
    #     interpreted_file = f"{input_dir}/{case_number}.csv"
    #     interpreted_file_with_sections = f"{sections_dir}/{case_number}.csv"
    #     processToGetTriples(llm_chain_extraction, interpreted_file_with_sections, interpreted_file_with_sections)
    #     time.sleep(10) # Sleep for 10 seconds to avoid rate limiting
    
   
    # with_phrases_path,without_phrases_path = get_the_final_files(sections_dir, output_folder_path_for_aggregated_result)
    # interpreted_file_with_phrases= f"{output_folder_path_for_aggregated_result}/ExplodedPhrases.csv" 
    # getTheLegitPhrases(with_phrases_path,interpreted_file_with_phrases)

        
  
    

if __name__ == "__main__":
    ref_list = [{'text': 'section 55', 'href': 'http://www.legislation.gov.uk/id/ukpga/1986/55/section/55', 'legislation_section': ('1986/55', '55')}, {'text': 'FLA 1986', 'href': 'http://www.legislation.gov.uk/id/ukpga/1986/55', 'legislation_section': ('1986/55', None)}, {'text': 'sections 31', 'href': 'http://www.legislation.gov.uk/id/ukpga/1984/42/section/31', 'legislation_section': ('1984/42', '31')}, {'text': 'Matrimonial and Family Proceedings Act 1984', 'href': 'http://www.legislation.gov.uk/id/ukpga/1984/42', 'legislation_section': ('1984/42', None)}]
    ref_list = [{'text': 'sections 31', 'href': 'http://www.legislation.gov.uk/id/ukpga/1984/42/section/31', 'legislation_section': ('1984/42', '31')}, {'text': 'Matrimonial and Family Proceedings Act 1984', 'href': 'http://www.legislation.gov.uk/id/ukpga/1984/42', 'legislation_section': ('1984/42', None)}]
    references = ['http://www.legislation.gov.uk/id/ukpga/1986/55','http://www.legislation.gov.uk/id/ukpga/1984/42']

    #result = get_all_the_sections_mentioned_explicitly(ref_list,'data/test2/csv_cases/legislation')
    text = '''Thus, in July 2021, Mr J made this application under  
	   section 55 A  FLA 1986 .  There was a regrettable, and largely unexplained, delay in progressing the case through the Family Court, and then in the purported (albeit erroneous 
		 ) transfer of the application by a circuit judge in the Family Court to the High Court.  A further short delay was occasioned while efforts were made, through the means of third party disclosure orders, to locate the mother and children.  That said, as soon as the mother was located and served, she promptly engaged with the process and filed her evidence as directed. 
	       Circuit Judges cannot transfer cases from the Family Court to the High Court: see  sections 31 I, 38 and 39 of the  Matrimonial and Family Proceedings Act 1984 ; rule 29.17(3) and (4) of the Family Procedure Rules 2010; President's Guidance on allocation and transfer (Feb 2018) ([26]).  The fact that no High Court Judge was involved in the purported transfer of the case to the Royal Courts of Justice  may  explain how the application disappeared into the system.'''
    result = get_the_relevant_sections(text,references,ref_list,'data/test2/csv_cases/legislation')
    print(result)
    print(len(result))
