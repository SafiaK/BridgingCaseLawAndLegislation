import os
print(os.getcwd())
print("================================")
from LegislationHandler_old import LegislationParser
import pandas as pd
from classifier import process_csv_with_openai
import time
import ast
import pickle
import keyPhraseExtractor
import util


#This is the main class for the pipeline
# Each step is listed and commented for clarity
# Step#1
##Convert_CSVs_xml_to_Csv
## The pipline starts with parsing the caselaw XML and converting it to Json format
# Step#2
## Extract_Judgment_Body_Paragraphs_Text
## Extracts the judgment body paragraphs and their corresponding text and references
# Step#3
### Extract_Legislation_Text
## Downlod the legislation act section by section




def downloadThelegislationIfNotExist(legislation_urls,legislation_dir):

    for url in legislation_urls:
        try:
            # Extract legislation ID from URL
            # Extract legislation path after ukpga
            # Split URL to get legislation path after ukpga
            if 'https' in url:
                legislation_parts = url.split('https://www.legislation.gov.uk/', 1)[-1].split('/')
            else:
                legislation_parts = url.split('http://www.legislation.gov.uk/', 1)[-1].split('/')
            
            
            # Create directory path preserving the full structure (e.g. Eliz2/8-9/65)
            directory_path = os.path.join(legislation_dir, *legislation_parts)
            
            # Check if the directory already exists and contains files
            if not os.path.exists(directory_path) or not os.listdir(directory_path):
                parser = LegislationParser(url, True)
                
                # Create directory if it does not exist
                os.makedirs(directory_path, exist_ok=True)
                
                parser.save_all_sections_to_files(directory_path)
                
        except Exception as e:
            print(f"Error processing legislation URL {url}: {str(e)}")
            continue

def extract_legislation_references(unprocessed_cases, folder_path):
        # Dictionary to store case number -> legislation URLs mapping
        case_legislation_map = {}
        
        for case in unprocessed_cases:
            legislation_urls = set()
            csv_path = f'{folder_path}/{case}.csv'
            print("------checking legislation for csv --------")
            print(csv_path)
            print(f"------------------------------------------")
            try:
                df = pd.read_csv(csv_path)
                # Extract legislation references from the references column
                for _, row in df.iterrows():
                    if not pd.isna(row['references']):
                        refs = ast.literal_eval(row['references'])
                        for ref in refs:
                            if 'href' in ref:
                                # Extract the legislation URI without section
                                uri = ref['href']

                                # Get just the legislation part before any section
                                #split the url on /section if it contains a section key words
                                
                                if '/section' in uri:
                                    bas_uri = uri.split('/section')[0]
                                    legislation_urls.add(bas_uri)
                                elif '/schedule' in uri:
                                    bas_uri = uri.split('/schedule')[0]
                                    legislation_urls.add(bas_uri)
                                elif '/regulation' in uri:
                                    bas_uri = uri.split('/regulation')[0]
                                    legislation_urls.add(bas_uri)
                                elif '/article' in uri:
                                    bas_uri = uri.split('/article')[0]
                                    legislation_urls.add(bas_uri)
                                elif '/chapter' in uri:
                                    bas_uri = uri.split('/chapter')[0]
                                    legislation_urls.add(bas_uri)
                                else:
                                    bas_uri = uri
                                    legislation_urls.add(bas_uri)
                    else:
                        print("reference column not found")


                
                # Add non-empty legislation URL lists to the map
                if legislation_urls:
                    case_legislation_map[case] = sorted(list(legislation_urls))
                    
            except Exception as e:
                print(f"Error processing case {case}: {e}")
                
        print(f"Found legislation references in {len(case_legislation_map)} cases")
        return case_legislation_map

def process_cases(input_folder_path, output_folder_path):
    #input_folder_path = a folder containing the XML files
    #input_folder_path =xml_to_csv = a folder to save the CSV files
    xml_to_csv = output_folder_path
    # Process only the cases in the issues list
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.xml'):
            # Extract case number from filename
            filename_case = filename.split('.')[0]
            case_number = filename_case.split('_')[-1]
            
            test_case_path = os.path.join(input_folder_path, filename)
            caselaw_csv_path = f'{xml_to_csv}/{case_number}.csv'
            print("================================")
            print(f"processing {caselaw_csv_path}")
            util.Convert_CSVs_xml_to_Csv(test_case_path, caselaw_csv_path)
def classify_the_paragraphs(xml_to_csv,examples_json_file_path):
    folder_path = xml_to_csv
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Extract case number from filename
            case_number = filename.split('.')[0]
            csv_path = os.path.join(folder_path, filename)
            print(csv_path)
            print(f"===========processing {case_number} =============================")
            process_csv_with_openai(examples_json_file_path, csv_path)
def process_legislation_references(folder_path, legislation_dir, input_folder_path):
    csv_files = [f.replace('.csv', '') for f in os.listdir(folder_path) if f.endswith('.csv')]
    case_legislation_map = extract_legislation_references(csv_files, folder_path)
    legislationlist = [item for sublist in case_legislation_map.values() for item in sublist]
    downloadThelegislationIfNotExist(legislationlist, legislation_dir)
    
    # Remove URL prefix from legislation references
    cleaned_case_legislation_map = {}
    for case, legislation_list in case_legislation_map.items():
        cleaned_legislation = []
        for legislation in legislation_list:    
            legislation_act =  legislation.replace('http://www.legislation.gov.uk/', '').replace('http://www.legislation.gov.uk/', '')          

            cleaned_legislation.append(legislation_act)
        cleaned_case_legislation_map[case] = cleaned_legislation
    
    print("Cleaned legislation references by case:")
    print(cleaned_case_legislation_map)
    
    # Save the cleaned legislation map to a pickle file
    with open(f'{input_folder_path}/cleaned_case_legislation_map.pkl', 'wb') as f:
        pickle.dump(cleaned_case_legislation_map, f)
    print("Saved cleaned legislation map to data/cleaned_case_legislation_map.pkl")
if __name__ == "__main__":
    
    '''
    input_folder_path = 'caselaw'
    output_folder_path = f"{input_folder_path}/output"
    os.makedirs(output_folder_path, exist_ok=True)
    xml_to_csv = f"{output_folder_path}/xml_to_csv"
    os.makedirs(xml_to_csv, exist_ok=True)
    legislation_dir = f"{input_folder_path}/legislation"
    os.makedirs(legislation_dir, exist_ok=True)
    
    ##########################STEP 1 PROCESSING###############################
    

    # Call the function
    process_cases(input_folder_path, xml_to_csv)
    
    ##########################STEP 2 Classifier###############################
    
    # Lists of case numbers - issues contains cases that had processing problems,
    
    
    classify_the_paragraphs(xml_to_csv)

    '''
    #######STEP 3 Attaching the sections with the paragraphs which are legally interpreted###############################
    #####part-1 step 3----download the legislation mentioned in the case laws########### 
    # Read all case CSVs and collect unique legislation references
    
    i = '036'
    xml_to_csv = f'data/final_test/case_csvs/cluster{i}'#xml_to_csv
    legislation_dir = 'data/final_test/case_csvs/legislation'
    input_folder_path = f'data/final_test/case_csvs/cluster{i}'
    output_folder_path = f'data/final_test/case_csvs/cluster{i}/output'
    


    # Call the function
    #process_legislation_references(xml_to_csv, legislation_dir, input_folder_path)
    
    
    #####part-2 step 3----make the Triples########### 
    pickle_file_path = f'data/final_test/case_csvs/cluster{i}/cluster{i}_cases.pkl'
    # Load and print the pickle file contents
    with open(pickle_file_path, 'rb') as f:
        pickle_data = pickle.load(f)
    
    print("Pickle file contents:")
    print(pickle_data)
    print(f"Type: {type(pickle_data)}")
    print(f"Length: {len(pickle_data) if hasattr(pickle_data, '__len__') else 'N/A'}")
    print(pickle_file_path)
    # Each item in pickle_data is a list of items -- compute the total number of unique items across all lists
    unique_items = []
    for item,values in pickle_data.items():
        if isinstance(values, list):
            unique_items.extend(values)
        
    print(f"Total unique items across all lists: {len(set(unique_items))}")
    keyPhraseExtractor.extractThePhrases(pickle_file_path,xml_to_csv,xml_to_csv,legislation_dir,output_folder_path)
