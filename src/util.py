
import re
from pathlib import Path
import JudgementHandler
import pandas as pd
import sys
import csv
import re
import json

def clean_dataframe_for_csv(df):
    """
    Clean DataFrame columns to prevent line breaks in CSV files.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Clean text columns
    text_columns = ['paragraphs', 'section_text', 'reason', 'application_of_law_phrases']
    
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('\n', ' ', regex=False)
            df_clean[col] = df_clean[col].str.replace('\r', ' ', regex=False)
            df_clean[col] = df_clean[col].str.replace('\t', ' ', regex=False)
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
            df_clean[col] = df_clean[col].str.strip()
    
    return df_clean

def get_base_url(uri):
    bas_uri = uri

    if '/section' in uri:
        bas_uri = uri.split('/section')[0]
    elif '/schedule' in uri:
        bas_uri = uri.split('/schedule')[0]
    elif '/regulation' in uri:
        bas_uri = uri.split('/regulation')[0]
    elif '/article' in uri:
        bas_uri = uri.split('/article')[0]
    elif '/chapter' in uri:
        bas_uri = uri.split('/chapter')[0]
    return bas_uri

def getTheFirstSection(ref_list,legislation_folder_path):
    for ref in ref_list:
        act,section = ref['legislation_section']
        if section:
            section_u = section.split('/')[0]
            with open(f'{legislation_folder_path}/{act}/section-{section_u}.txt', 'r') as file:
                content = file.read()
                return content
    return ''

def flatten_list_of_lists(list_of_lists):
    """
    Flattens a list of lists into a single list containing all the values.

    Args:
        list_of_lists (list): A list where each element is a list.

    Returns:
        list: A single list containing all the values from the input lists.
    """
    return [item for sublist in list_of_lists for item in sublist]
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
def is_all_stopwords(phrase, custom_stopwords_file):
    """
    Check if all words in a phrase are stopwords (either in custom list or English stopwords).
    
   Args:
        phrase (str): The phrase to check.
        custom_stopwords_file (str): Path to file containing custom stopwords.
    
    Returns:
        bool: True if all words are stopwords, False otherwise
    """
    # Convert custom stopwords to a lowercase set for efficient lookup
    # Read custom stopwords from file
    with open(custom_stopwords_file, 'r') as f:
        custom_stopwords = set(word.strip().lower() for word in f.readlines())
    #print(custom_stopwords)
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Split phrase into words and convert to lowercase
    phrase_words = phrase.lower().split()
    #print(f"Phrase words: {phrase_words}")
    
    # Check if all words in the phrase are stopwords
    for word in phrase_words:
        # Get all variations of the word
        variations = {
            word,  # Original lowercase
            stemmer.stem(word),  # Stemmed
            lemmatizer.lemmatize(word)  # Lemmatized
        }
        #print(f"Variations for '{word}': {variations}")
        
        # Check if any variation is in custom or NLTK stopwords
        if not any(var in custom_stopwords or var in STOP_WORDS for var in variations):
            return False
            
    return True
import requests
import xml.etree.ElementTree as ET
import os
from datetime import datetime



def fetch_atom_feed(page=1):
    BASE_URL = "https://caselaw.nationalarchives.gov.uk/atom.xml"
    """Fetches a specific page from the Atom XML feed."""
    url = f"{BASE_URL}?page={page}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return ET.ElementTree(ET.fromstring(response.content))
    else:
        print(f"⚠️ Failed to fetch Atom feed (Page {page}). Status Code: {response.status_code}")
        return None
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
def extract_year_from_text(text):
    """Extracts the case decision year from a text field using regex."""
    match = re.search(r'\b(20\d{2}|19\d{2})\b', text)  # Find year (e.g., 2005, 2012, 2020)
    if match:
        return int(match.group(1))  # Convert to integer
    return None  # No valid year found
def parse_caselaws(tree,START_YEAR,END_YEAR):
    """Extracts tribunal caselaws within the specified year range."""
    root = tree.getroot()
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall(".//atom:entry", namespace)

    case_links = []
    
    for entry in entries:
        try:
            title = entry.find("atom:title", namespace).text.strip()
            #updated = entry.find("atom:updated", namespace).text.strip()
            link_element = entry.find("atom:link[@rel='alternate']", namespace)

            # Extract case decision year from summary or other fields
            case_year = None
         
            if link_element is not None:
                case_url = link_element.attrib.get("href")
                case_year = extract_year_from_url(case_url)  # Extract from URL


                if START_YEAR <= case_year <= END_YEAR:
                    case_links.append((title, case_url, case_year))
        except Exception as e:
            print(f"�� The name of the title and url of caselaw is missing: {e}")

    return case_links

def sanitize_filename(filename):
    """Replaces invalid characters for filenames."""
    invalid_chars = r'\/:*?"<>|'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename
def extract_year_from_url(case_url):
    """Extracts the case decision year from the URL."""
    match = re.search(r'/(\d{4})/', case_url)  # Looks for YYYY in the URL
    if match:
        return int(match.group(1))  # Convert to integer
    return None  # No valid year found
def download_xml_case_given_the_url(case_url, SAVE_DIR):
    try:
        xml_url = case_url + "/data.xml"
        response = requests.get(xml_url)

        if response.status_code == 200:
            save_path = os.path.join(SAVE_DIR)
            os.makedirs(save_path, exist_ok=True)

            # Clean up filename
            title = case_url.split("https://caselaw.nationalarchives.gov.uk/")[1].replace("/", "_")
            file_name = sanitize_filename(title) + ".xml"
            file_path = os.path.join(save_path, file_name)

            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"✅ Saved: {file_path}")
        else:
            print(f"❌ Failed to download XML for {title} - {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading {title}: {e}")
def download_xml_cases(case_links):
    """Downloads case law XML files and saves them in structured folders."""
    for title, case_url, case_year in case_links:
        try:
            xml_url = case_url + "/data.xml"
            response = requests.get(xml_url)

            if response.status_code == 200:
                save_path = os.path.join(SAVE_DIR, str(case_year))
                os.makedirs(save_path, exist_ok=True)

                # Clean up filename
                file_name = sanitize_filename(title) + ".xml"
                file_path = os.path.join(save_path, file_name)

                with open(file_path, "wb") as f:
                    f.write(response.content)

                print(f"✅ Saved: {file_path}")
            else:
                print(f"❌ Failed to download XML for {title} - {response.status_code}")
        except Exception as e:
            print(f"❌ Error downloading {title}: {e}")

def fetch_all_pages(start_page=1, START_YEAR=2000, END_YEAR=2025,end_page=827):
    """Iterates through all Atom feed pages and downloads all tribunal caselaws from 2000-2025."""
    
    page = start_page
    total_cases = 0

    while True:
        try:
            print(f"📡 Fetching Page {page}...")
            tree = fetch_atom_feed(page)

            if not tree:
                break  # Stop if no more pages
            # Save the XML tree to a file for inspection
            tree.write(f"atom_feed_page_{page}.xml")

            # Print all elements of the tree for debugging
            for elem in tree.iter():
                print(elem.tag, elem.attrib)

            case_links = parse_caselaws(tree,START_YEAR,END_YEAR)
            #print(case_links)

            if not case_links:
                print(f"⚠️ No more tribunal caselaws found on Page {page}. Stopping.")
                break  # No more cases found → exit loop

            print(f"📂 Found {len(case_links)} cases on Page {page}. Downloading...")
            download_xml_cases(case_links)
            total_cases += len(case_links)
            
            page += 1  # Move to the next page
            if page > end_page:
                break  # Stop if the end page is reached
        except Exception as e:
            print(f"�� Error fetching or parsing cases: {e}")
            break  # Stop if an error occurs while fetching or parsing cases
            

    print(f"✅ Download complete! {total_cases} cases saved.")
def count_xml_files_in_caselaw(caselaw_folder="caselaw"):
    """
    Iterates through each subfolder in the given 'caselaw' folder and computes
    the number of XML files contained (recursively) within each subfolder.

    Args:
        caselaw_folder (str): The path to the base folder (default "caselaw").

    Returns:
        dict: A dictionary where each key is the subfolder name and the value is the count of XML files.
    """
    counts = {}
    # List immediate subdirectories in the caselaw folder
    for subfolder in os.listdir(caselaw_folder):
        subfolder_path = os.path.join(caselaw_folder, subfolder)
        if os.path.isdir(subfolder_path):
            xml_count = 0
            # Walk through the subfolder recursively
            for dirpath, dirnames, filenames in os.walk(subfolder_path):
                for file in filenames:
                    if file.lower().endswith(".xml"):
                        xml_count += 1
            counts[subfolder] = xml_count
    return counts

def validate_para_ids(csv_file_path):
    """
    Validates that each paragraph in the CSV file starts with its corresponding para_id.

    Parameters:
    csv_file_path (str): The path to the CSV file.

    Returns:
    bool: True if all para_id values are legitimate, False otherwise.
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            para_id = row['para_id'].strip()
            paragraph = row['paragraphs'].strip()

            # Extract the numerical part of para_id
            match = re.match(r'para_(\d+)', para_id)
            if match:
                para_number = match.group(1)
                # Check if paragraph starts with the same number
                if not paragraph.startswith(f"{para_number}."):
                    return False
            else:
                return False

    return True


def Convert_CSVs_xml_to_Csv(caselaw_xml_path,caselaw_csv_path,convert_if_legislation=True):

    def create_and_save_dataframe_with_data(file_path, data1, data2,data3,data4):
        # Set default column names if none are provided
        
        col_names = ["case_uri","para_id","paragraphs","references"]
        
        # Extract the file name without extension from the path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create a DataFrame with the provided data
        data = pd.DataFrame({col_names[0]: data1, col_names[1]: data2,col_names[2]:data3,col_names[3]:data4})
        
        # Save the DataFrame as a CSV file with the extracted name
        csv_filename = f"{base_name}.csv"
        data.to_csv(caselaw_csv_path, index=False)
        
        return csv_filename
    parser = JudgementHandler.JudgmentParser(caselaw_xml_path)
    #Only those cases are reffered which have type=legislation as a reference
    if convert_if_legislation:
        if parser.has_legislation_reference():
            judgment_body_paragraphs_text = parser.get_judgment_body_paragraphs_text()
            
            para_contents = []
            para_ids = [] 
            para_references = []
            case_uris = []
            for para in judgment_body_paragraphs_text: 
                caseuri, para_id, content, references = para
                para_contents.append(content)
                para_ids.append(para_id)
                para_references.append(references)
                case_uris.append(caseuri)
            create_and_save_dataframe_with_data(caselaw_csv_path, case_uris, para_ids, para_contents, para_references)
    else:
        judgment_body_paragraphs_text = parser.get_judgment_body_paragraphs_text()
        
        para_contents = []
        para_ids = [] 
        para_references = []
        case_uris = []
        for para in judgment_body_paragraphs_text: 
            caseuri, para_id, content, references = para
            para_contents.append(content)
            para_ids.append(para_id)
            para_references.append(references)
            case_uris.append(caseuri)
        create_and_save_dataframe_with_data(caselaw_csv_path, case_uris, para_ids, para_contents, para_references)

    

def convert_all_xml_to_csv_report(caselaw_base_folder: str = "caselaw", csv_output_base: str = "caselaw_csv") -> dict:
    """
    Converts all XML files in each yearly subfolder of the caselaw folder to CSV,
    but only converts those XML files that have a legislation reference.
    
    For each year folder under caselaw_base_folder, the function scans for XML files.
    If an XML file contains a legislation reference (as determined by JudgmentParser),
    it is converted to CSV using the existing Convert_CSVs_xml_to_Csv function and saved 
    in a corresponding subfolder under csv_output_base.
    
    Returns:
        dict: A report dictionary where each key is a year (folder name) with the count 
              of converted caselaws, plus a "total" key for the overall count.
    
    Example folder structure:
      caselaw/
        2001/
          case1.xml, case2.xml, ...
        2002/
          case3.xml, ...
      csv_output_base/
        2001/
          case1.csv, case2.csv, ...
        2002/
          case3.csv, ...
    """
    caselaw_path = Path(caselaw_base_folder)
    csv_output_path = Path(csv_output_base)
    csv_output_path.mkdir(exist_ok=True)
    
    conversion_report = {}
    total_converted = 0
    
    # Iterate through each subfolder (year)
    for year_folder in caselaw_path.iterdir():
        if year_folder.is_dir():
            year = year_folder.name
            year_csv_folder = csv_output_path / year
            year_csv_folder.mkdir(exist_ok=True)
            count = 0
            
            # Process each XML file in the year folder
            for xml_file in year_folder.glob("*.xml"):
                csv_file = year_csv_folder / f"{xml_file.stem}.csv"
                try:
                    # Call the conversion function (assumed to be defined elsewhere)
                    Convert_CSVs_xml_to_Csv(str(xml_file), str(csv_file))
                    if csv_file.exists():
                        count += 1
                except Exception as e:
                    print(f"Error converting {xml_file}: {e}")
            conversion_report[year] = count
            total_converted += count
    
    conversion_report["total"] = total_converted
    return conversion_report
def change_csv_file_name(csv_file_path):
    """
    Changes the name of a CSV file to a new name.

    Parameters:
    csv_file_path (str): The path to the original CSV file.
    new_name (str): The new name for the CSV file (without extension).
    """

    #read the csv file in pandas dataframe
    df = pd.read_csv(csv_file_path)
    # Extract the case_uri and para_id from the first row
    case_uri = df['case_uri'].iloc[0]
    #from     case_uri remove 'https://caselaw.nationalarchives.gov.uk/' and replace '/' with '_'
    filr_name = case_uri.replace('https://caselaw.nationalarchives.gov.uk/', '').replace('/', '_')
    # Save the csv file with the new name
    directory = os.path.dirname(csv_file_path)
    new_file_path = os.path.join(directory, filr_name + '.csv')
    df.to_csv(new_file_path, index=False)
    os.remove(csv_file_path)  # Remove the old file
    # Ensure the new file name is valid


 
    # Extract the directory and file name from the original path
def process_caselaws_directory(base_dir):
    """
    Processes CSV files in the Caselaws_CSV directory and deletes those that don't meet the validate_para_ids criteria.

    Parameters:
    base_dir (str): The path to the Caselaws_CSV directory.
    """
    # Iterate over each subdirectory and file in the base directory
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                if not validate_para_ids(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted invalid file: {file_path}")
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")

def count_csv_files_in_subdirectories(base_dir):
    """
    Counts the number of CSV files in each subdirectory of the given base directory
    and prints the results in a report format.

    Parameters:
    base_dir (str): The path to the base directory containing subdirectories.
    """
    # Print header
    print(f"{'Subdirectory':<50} {'CSV File Count'}")
    print("=" * 60)

    # Iterate over each subdirectory in the base directory
    for root, dirs, files in os.walk(base_dir):
        # Skip the base directory itself
        if root == base_dir:
            continue
        # Filter out CSV files
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        # Count the number of CSV files
        csv_count = len(csv_files)
        # Print the subdirectory path and CSV count
        print(f"{root:<50} {csv_count}")




if __name__ == "__main__":
    
    count_csv_files_in_subdirectories("Caselaws_CSV")
    exit(1)
    # Example usage:
    process_caselaws_directory("Caselaws_CSV")
    
    
    report = convert_all_xml_to_csv_report("caselaw", "caselaw_csv")
    print("Conversion Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    exit(1)
    # Base URL for Atom feed
    SAVE_DIR = "caselaw"  # Directory to save case law XML files
    START_YEAR = 2000
    END_YEAR = 2025
    #fetch_all_pages(start_page=1, START_YEAR=START_YEAR, END_YEAR=END_YEAR,end_page=2000)
    xml_counts = count_xml_files_in_caselaw(SAVE_DIR)
    for subfolder, count in xml_counts.items():
        print(f"{subfolder}: {count} XML files")



