import pandas as pd
from classifier import process_csv_with_openai
def process_single_csv(csv_path, model):
    """
    Process a single CSV file with a specific language model and save results back to the same file.
    
    Args:
        csv_path (str): Path to the CSV file to process
        model (str): Name of the language model to use
    """
    # Read the CSV file and rename initial columns
    df = pd.read_csv(csv_path)
    if 'if_law_applied' in df.columns:
        df.rename(columns={'if_law_applied': 'if_law_applied_actual'}, inplace=True)
    if 'application_of_law_phrases' in df.columns:
        df.rename(columns={'application_of_law_phrases': 'application_of_law_phrases_actual'}, inplace=True)
    
    df.to_csv(csv_path, index=False)

    print(f"=========== Processing with {model} =============================")
    delay = 0
    batch_size = 20
    max_tokens = 500000
    
    if model == 'deepseek-8b':
        batch_size = 5
        max_tokens = 20000
    else:
        batch_size = 30

    process_csv_with_openai('examples.json', csv_path, model, batch_size, delay)
    
    # Read and rename columns with model suffix
    df = pd.read_csv(csv_path)
    df.rename(columns={
        'if_law_applied': f'if_law_applied_{model}',
        'application_of_law_phrases': f'application_of_law_phrases_{model}',
        'reason_of_choosing_it_as_application': f'reason_{model}'
    }, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"Done processing with {model}")

if __name__ == "__main__":
    csv_path = "ukftt_grc_2025_289.csv"
    models = ['deepseek-qwen-32b']
    
    for model in models:
        process_single_csv(csv_path, model)