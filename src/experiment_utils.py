import os
import json
import pandas as pd
import numpy as np
import pickle
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# Import project modules
from openAIHandler import getLegalClassifierChain, getPhraseExtractionChain, getInterPretations
from JudgementHandler import JudgmentParser
from util import Convert_CSVs_xml_to_Csv
from classifier import process_csv_with_openai
import keyPhraseExtractor

def load_csv_files(folder_path):
    """Load all CSV files from a folder into a dictionary of DataFrames."""
    csv_files = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            case_name = file.replace('.csv', '')
            try:
                csv_files[case_name] = pd.read_csv(file_path)
                print(f"Loaded {case_name} with {len(csv_files[case_name])} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    return csv_files

def prepare_training_data(case_files, test_case_name, false_positives_df, false_negatives_df,all_cases_data_dict):
    """
    Prepare training data for a specific test case.
    Use all other cases as training examples, prioritizing examples marked wrong by most LLMs.
    
    Args:
        case_files (dict): Dictionary of case files (name -> DataFrame)
        test_case_name (str): Name of the test case to exclude from training
        false_positives_df (DataFrame): DataFrame of false positives
        false_negatives_df (DataFrame): DataFrame of false negatives
        
    Returns:
        list: List of training examples
    """
    # Initialize training data
    training_examples = []
    
   
    # First, add false positives and false negatives from other cases
    if not false_positives_df.empty:
        for _, row in false_positives_df.iterrows():
            case_name = row.get('case_uri', '').split('https://caselaw.nationalarchives.gov.uk/')[1].replace('/','_')
            
            para_id = case_name +'_'+row.get('para_id', '').replace('para_','')
            #all_cases_data is a list
            #find the para_id if in all_cases_data list
            #then pick up the reason from there

            reason = ''
            if all_cases_data_dict.get(para_id,0):
                obj = all_cases_data_dict[para_id]
                print(all_cases_data_dict[para_id])

                reason = all_cases_data_dict[para_id]['reason']
            


            if case_name != test_case_name:
                training_examples.append({
                    "para_id": case_name +'_'+row.get('para_id', ''),
                    "para_content": row.get('paragraphs', ''),
                    "if_law_applied": False,  # False positive becomes a negative example
                    "application_of_law_phrases": [],
                    "reason": reason
                })
    
    if not false_negatives_df.empty:
        for _, row in false_negatives_df.iterrows():
            case_name = row.get('case_uri', '').split('https://caselaw.nationalarchives.gov.uk/')[1].replace('/','_')

            reason = ''
            if all_cases_data_dict.get(para_id,0):
                reason = all_cases_data_dict[para_id]['reason']

            if case_name != test_case_name:
                training_examples.append({
                    "para_id": case_name +'_'+row.get('para_id', ''),
                    "para_content": row.get('paragraphs', ''),
                    "if_law_applied": True,  # False negative becomes a positive example
                    "application_of_law_phrases": row.get('application_of_law_phrases_actual', ''),
                    "reason": reason
                })
    
    # Then, add examples from other cases, prioritizing those marked wrong by most LLMs
    for case_name, df in case_files.items():
        if case_name != test_case_name:
            # Count positive examples to balance later
            positive_count = 0
            
            # First, add examples that were marked wrong by most LLMs
            if 'if_law_applied_actual' in df.columns and 'if_law_applied_claude' in df.columns and 'if_law_applied_gpt4' in df.columns:
                for _, row in df.iterrows():
                    actual = row.get('if_law_applied_actual', 0)
                    claude = row.get('if_law_applied_claude', 0)
                    gpt4 = row.get('if_law_applied_gpt4', 0)
                    gpt4_mini = row.get('if_law_applied_gpt4_mini', 0)
                    llama = row.get('if_law_applied_llama', 0)
                    
                    # Count how many LLMs got it wrong
                    wrong_count = 0
                    if claude != actual:
                        wrong_count += 1
                    if gpt4 != actual:
                        wrong_count += 1
                    if gpt4_mini != actual:
                        wrong_count += 1
                    if llama != actual:
                        wrong_count += 1
                    
                    # If most LLMs got it wrong, add it as a priority example
                    if wrong_count >= 2:
                        if actual == 1:
                            positive_count += 1
                        reason = ''
                        print(para_id)
                        if all_cases_data_dict.get(para_id,0):
                            reason = all_cases_data_dict[para_id]['reason']
                        training_examples.append({
                            "para_id": case_name +'_'+ row.get('para_id', ''),
                            "para_content": row.get('paragraphs', ''),
                            "if_law_applied": actual == 1,
                            "application_of_law_phrases": row.get('application_of_law_phrases_actual', '') if actual == 1 else [],
                            "reason": reason
                        })
    
    # Balance the dataset by adding more examples if needed
    positive_examples = [ex for ex in training_examples if ex["if_law_applied"]]
    negative_examples = [ex for ex in training_examples if not ex["if_law_applied"]]
    
    print(f"Initial training data: {len(positive_examples)} positive, {len(negative_examples)} negative")
    
    # If we need more examples to balance the dataset
    if len(positive_examples) < len(negative_examples):
        # Add more positive examples from other cases
        needed = len(negative_examples) - len(positive_examples)
        added = 0
        
        for case_name, df in case_files.items():
            if case_name != test_case_name and added < needed:
                if 'if_law_applied_actual' in df.columns:
                    for _, row in df.iterrows():
                        if row.get('if_law_applied_actual', 0) == 1:
                            # Check if this example is already in training_examples
                            para_id = row.get('para_id', '')
                            if not any(ex["para_id"] == para_id for ex in training_examples):
                                reason = ''
                                print(para_id)
                                if all_cases_data_dict.get(para_id,0):
                                    reason = all_cases_data_dict[para_id]['reason']
                                training_examples.append({
                                    "para_id": case_name +'_'+para_id,
                                    "para_content": row.get('paragraphs', ''),
                                    "if_law_applied": True,
                                    "application_of_law_phrases": row.get('application_of_law_phrases_actual', ''),
                                    "reason": reason
                                })
                                added += 1
                                if added >= needed:
                                    break
    elif len(negative_examples) < len(positive_examples):
        # Add more negative examples from other cases
        needed = len(positive_examples) - len(negative_examples)
        added = 0
        
        for case_name, df in case_files.items():
            if case_name != test_case_name and added < needed:
                if 'if_law_applied_actual' in df.columns:
                    for _, row in df.iterrows():
                        if row.get('if_law_applied_actual', 0) == 0:
                            # Check if this example is already in training_examples
                            para_id = row.get('para_id', '')
                            if not any(ex["para_id"] == para_id for ex in training_examples):
                                
                                training_examples.append({
                                    "para_id": para_id,
                                    "para_content": row.get('paragraphs', ''),
                                    "if_law_applied": False,
                                    "application_of_law_phrases": [],
                                    "reason": f"Additional negative example from {case_name}"
                                })
                                added += 1
                                if added >= needed:
                                    break
    
    # Final count
    positive_examples = [ex for ex in training_examples if ex["if_law_applied"]]
    negative_examples = [ex for ex in training_examples if not ex["if_law_applied"]]
    print(f"Final training data: {len(positive_examples)} positive, {len(negative_examples)} negative")
    
    return training_examples

def run_experiment(test_case_name, model_name, training_data_path, input_folder_path, experiment_folder_path):
    """
    Run an experiment for a specific test case and model.
    
    Args:
        test_case_name (str): Name of the test case
        model_name (str): Name of the model to use
        training_data_path (str): Path to the training data directory
        input_folder_path (str): Path to the input folder
        experiment_folder_path (str): Path to the experiment output folder
        
    Returns:
        dict: Experiment results including metrics
    """
    print(f"Running experiment for {test_case_name} with {model_name}")
    
    # Load training data
    training_file_path = os.path.join(training_data_path, f"{test_case_name}_training.json")
    
    
    # Load test data
    test_file_path = os.path.join(input_folder_path, f"{test_case_name}.csv")
    test_df = pd.read_csv(test_file_path)
    
    if 'if_law_applied' in test_df.columns:
        test_df.rename(columns={'if_law_applied': 'if_law_applied_actual'}, inplace=True)
    if 'application_of_law_phrases' in test_df.columns:
        test_df.rename(columns={'application_of_law_phrases': 'application_of_law_phrases_actual'}, inplace=True)
    
    test_df.to_csv(test_file_path, index=False)
    print(f"===========processing {test_case_name} =============================")
    delay = 0
    batch_size = 20
    max_tokens = 500,000
    #if model == 'llama-3.3-70b-versatile': # for free toer
        # delay = 60
    #else:
        #delay = 0
    if model_name == 'claude-3-7-sonnet-latest':
        batch_size = 5
        max_tokens = 20,000
    else:
        batch_size = 30
        #max_tokens = 500,000
    process_csv_with_openai(training_file_path, test_file_path, model_name,batch_size,delay)
    df = pd.read_csv(test_file_path)
    df.rename(columns={
        'if_law_applied': f'if_law_applied_{model_name}', 
        'application_of_law_phrases': f'application_of_law_phrases_{model_name}',
        'reason_of_choosing_it_as_application': f'reason_{model_name}'
    }, inplace=True)
    df.to_csv(test_file_path, index=False)
    print("Done")


    '''



    
    # Create output path
    #output_file_path = os.path.join(experiment_folder_path, f"{test_case_name}_{model_name}_result.csv")
    output_file_path = os.path.join(experiment_folder_path, f"{test_case_name}.csv")
    
    # Run the classifier chain
    #try:
    #parser, chain = getLegalClassifierChain(training_file_path, model_name)
    process_csv_with_openai(training_file_path, test_file_path, model_name)
    
    # Load results
    result_df = pd.read_csv(output_file_path)
    '''
    # Calculate metrics
    metrics = calculate_metrics(df,model_name)
    
    return {
        "test_case": test_case_name,
        "model": model_name,
        "metrics": metrics
    }
    # except Exception as e:
    #     print(f"Error running experiment: {str(e)}")
    #     return {
    #         "test_case": test_case_name,
    #         "model": model_name,
    #         "error": str(e)
    #     }
    

def calculate_metrics(actual_df, model_name):
    """
    Calculate precision, recall, F1 score, and accuracy for the experiment.
    
    Args:
        actual_df (pandas.DataFrame): DataFrame with actual values
        predicted_df (pandas.DataFrame): DataFrame with predicted values
        
    Returns:
        dict: Metrics including precision, recall, F1 score, and accuracy
    """


    # Convert values to binary
    y_true = actual_df['if_law_applied_actual'].astype(int).values
    
    # Convert predicted values to binary (they might be strings or booleans)
    col_name = f'if_law_applied_{model_name}'
    print(actual_df.columns)

    if actual_df[col_name].dtype == bool:
        y_pred = actual_df[col_name].astype(int).values
    else:
        # Try to convert strings like '1', '0', 'True', 'False' to integers
        y_pred = []
        for val in actual_df[col_name]:
            if isinstance(val, (int, float, np.number)):
                y_pred.append(int(val))
            elif isinstance(val, str):
                if val.lower() in ['1', 'true', 'yes']:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            elif isinstance(val, bool):
                y_pred.append(int(val))
            else:
                y_pred.append(0)  # Default to 0 for unknown values
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def generate_experiment_report(results):
    """
    Generate a report of experiment results.
    
    Args:
        results (list): List of experiment results
        
    Returns:
        pandas.DataFrame: DataFrame with experiment results
    """
    # Create a list to store report data
    report_data = []
    
    # Process each result
    for result in results:
        if "error" in result:
            # Skip results with errors
            continue
        
        # Extract metrics
        metrics = result.get("metrics", {})
        
        # Add to report data
        report_data.append({
            "Test Case": result["test_case"],
            "Model": result["model"],
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1 Score": metrics.get("f1", 0),
            "Accuracy": metrics.get("accuracy", 0),
            "True Positives": metrics.get("true_positives", 0),
            "False Positives": metrics.get("false_positives", 0),
            "True Negatives": metrics.get("true_negatives", 0),
            "False Negatives": metrics.get("false_negatives", 0)
        })
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Calculate average metrics by model
    avg_by_model = report_df.groupby("Model")[["Precision", "Recall", "F1 Score", "Accuracy"]].mean().reset_index()
    avg_by_model["Test Case"] = "Average"
    
    # Add average metrics to the report
    report_df = pd.concat([report_df, avg_by_model])
    
    return report_df

def visualize_results(report_df, experiment_folder_path):
    """
    Create visualizations of experiment results.
    
    Args:
        report_df (pandas.DataFrame): DataFrame with experiment results
        experiment_folder_path (str): Path to save visualizations
    """
    # Create visualizations directory
    viz_dir = os.path.join(experiment_folder_path, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize F1 scores by model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="F1 Score", data=report_df[report_df["Test Case"] == "Average"])
    plt.title("Average F1 Score by Model")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "f1_score_by_model.png"))
    plt.close()
    
    # Visualize F1 scores by test case and model
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Test Case", y="F1 Score", hue="Model", data=report_df[report_df["Test Case"] != "Average"])
    plt.title("F1 Score by Test Case and Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "f1_score_by_case_model.png"))
    plt.close()

def process_legislation_references(folder_path, legislation_dir, input_folder_path):
    """
    Process legislation references from case files.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        legislation_dir (str): Path to the legislation directory
        input_folder_path (str): Path to save the cleaned legislation map
    """
    csv_files = [f.replace('.csv', '') for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    case_legislation_map = extract_legislation_references(csv_files, folder_path)
    
    legislationlist = [item for sublist in case_legislation_map.values() for item in sublist]
    
    downloadThelegislationIfNotExist(legislationlist, legislation_dir)
    
    # Remove URL prefix from legislation references
    cleaned_case_legislation_map = {}
    for case, legislation_list in case_legislation_map.items():
        cleaned_legislation = []
        for legislation in legislation_list:    
            legislation_act = legislation.replace('http://www.legislation.gov.uk/', '').replace('http://www.legislation.gov.uk/', '')          
            cleaned_legislation.append(legislation_act)
        cleaned_case_legislation_map[case] = cleaned_legislation
    
    print("Cleaned legislation references by case:")
    print(cleaned_case_legislation_map)
    
    # Save the cleaned legislation map to a pickle file
    with open(f'{input_folder_path}/cleaned_case_legislation_map.pkl', 'wb') as f:
        pickle.dump(cleaned_case_legislation_map, f)
    print(f"Saved cleaned legislation map to {input_folder_path}/cleaned_case_legislation_map.pkl")

def extract_legislation_references(case_names, folder_path):
    """
    Extract legislation references from case files.
    
    Args:
        case_names (list): List of case names
        folder_path (str): Path to the folder containing CSV files
        
    Returns:
        dict: Dictionary mapping case names to legislation references
    """
    case_legislation_map = {}
    
    for case_name in case_names:
        file_path = os.path.join(folder_path, f"{case_name}.csv")
        try:
            df = pd.read_csv(file_path)
            legislation_refs = set()
            
            # Extract legislation references from the references column
            if 'references' in df.columns:
                for _, row in df.iterrows():
                    if not pd.isna(row['references']):
                        refs = ast.literal_eval(row['references'])
                        for ref in refs:
                            if 'href' in ref:
                                # Extract the legislation URI without section
                                uri = ref['href']
                                
                                # Get just the legislation part before any section
                                for keyword in ['section', 'schedule', 'regulation', 'article', 'chapter']:
                                    if f'/{keyword}' in uri:
                                        base_uri = uri.split(f'/{keyword}')[0]
                                        legislation_refs.add(base_uri)
                                        break
                                else:
                                    # If no keyword found, add the full URI
                                    legislation_refs.add(uri)
            
            case_legislation_map[case_name] = list(legislation_refs)
        except Exception as e:
            print(f"Error extracting legislation references from {case_name}: {str(e)}")
            case_legislation_map[case_name] = []
    
    return case_legislation_map

def downloadThelegislationIfNotExist(legislation_urls, legislation_dir):
    """
    Download legislation if it doesn't exist.
    
    Args:
        legislation_urls (list): List of legislation URLs
        legislation_dir (str): Path to the legislation directory
    """
    os.makedirs(legislation_dir, exist_ok=True)
    
    for url in legislation_urls:
        # Extract legislation ID from URL
        legislation_id = url.split('/')[-1]
        file_path = os.path.join(legislation_dir, f"{legislation_id}.xml")
        
        # Check if file already exists
        if not os.path.exists(file_path):
            try:
                print(f"Downloading legislation: {url}")
                # Use requests to download the legislation
                import requests
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {legislation_id}")
                else:
                    print(f"Failed to download {url}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
        else:
            print(f"Legislation {legislation_id} already exists")

def measure_phrase_extraction_accuracy(actual_file, predicted_file):
    """
    Measure the accuracy of phrase extraction.
    
    Args:
        actual_file (str): Path to the file with actual phrases
        predicted_file (str): Path to the file with predicted phrases
        
    Returns:
        dict: Metrics including precision, recall, and F1 score
    """
    try:
        actual_df = pd.read_csv(actual_file)
        predicted_df = pd.read_csv(predicted_file)
        
        # Merge dataframes on para_id
        merged_df = pd.merge(actual_df, predicted_df, on='para_id', suffixes=('_actual', '_predicted'))
        
        # Count matches
        total_actual = 0
        total_predicted = 0
        total_matched = 0
        
        for _, row in merged_df.iterrows():
            if pd.notna(row.get('application_of_law_phrases_actual')) and pd.notna(row.get('triples_result')):
                actual_phrases = ast.literal_eval(row['application_of_law_phrases_actual']) if isinstance(row['application_of_law_phrases_actual'], str) else []
                predicted_phrases = ast.literal_eval(row['triples_result']) if isinstance(row['triples_result'], str) else []
                
                total_actual += len(actual_phrases)
                total_predicted += len(predicted_phrases)
                
                # Count matches (exact or partial)
                for actual_phrase in actual_phrases:
                    for predicted_phrase in predicted_phrases:
                        if 'case_law_term' in predicted_phrase and actual_phrase in predicted_phrase['case_law_term']:
                            total_matched += 1
                            break
        
        # Calculate metrics
        precision = total_matched / total_predicted if total_predicted > 0 else 0
        recall = total_matched / total_actual if total_actual > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_actual": total_actual,
            "total_predicted": total_predicted,
            "total_matched": total_matched
        }
    except Exception as e:
        print(f"Error measuring phrase extraction accuracy: {str(e)}")
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "error": str(e)
        }
import json
from openAIHandler import getLegalClassifierUsingJson
def run_full_case_experiment(case_json_path, training_examples, output_path):
    """
    Run an experiment with a full case law.
    
    Args:
        case_json_path (str): Path to the case JSON file
        training_examples (list): List of training examples
        model_name (str): Name of the model to use
        output_path (str): Path to save the output
        
    Returns:
        dict: Experiment results
    """
    try:
        # Load case JSON
        with open(case_json_path, 'r') as f:
            case_data = json.load(f)
        
        
        
        
        #print(examples)
        response = getLegalClassifierUsingJson(case_data,training_examples)
        print(response)
        response_updated = response.content.replace("```json\n","")
        response_updated = response_updated.replace("\n```","")
        #put response_updated in a json file
        with open(output_path, 'w') as f:
            f.write(response_updated)
        
        
        # Parse the response
        response_json = json.loads(response_updated)
        
        # Convert to DataFrame
        df = pd.DataFrame(response_json)
        
        # Save to CSV
        df.to_csv(output_path.replace('.json','.csv'), index=False)
    except Exception as e:
        print(f"Error running full case experiment: {str(e)}")
        return {
            "error": str(e)
        }

    
