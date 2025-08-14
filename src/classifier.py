import pandas as pd
from openAIHandler import getLegalClassifierChain
import ast
import json
import time
import util
def getExamples(file_path):

    def getTheParaId(case_uri, para_id):
        try:
            # Ensure para_id is converted to a string for consistent processing
            para = str(para_id)
            
            # Extract case number from case_uri
            case_number = case_uri.split("/")[-1]
            
            # Split para_id on "_" and get the second part (index 1), if it exists
            para_parts = para.split("_")
            
            # Ensure there are at least two parts to avoid IndexError
            if len(para_parts) > 1:
                para_number = para_parts[1]
            else:
                raise ValueError("Invalid para_id format, expecting an underscore.")

            # Return the formatted result
            return f"{case_number}_{para_number}"
        
        except Exception as e:
            print(f"Error processing case_uri: {case_uri}, para_id: {para_id}. Error: {e}")
            return None  # Return None or another placeholder if there's an error
    def getTheExampleTuple(row):
        try:
            pharses = []
            #print(type(row['label']))
            if not pd.isna(row['label']):
                labels = ast.literal_eval(row['label'])
                for label in labels:
                    pharses.append(label['text'])
            exampleDic = {'id':row['id'],
            'para_content' : row['ProcessedParagraphs'],
            'if_interpretation' : row['classifier_label'],
            'interpreted_phrases' :pharses}
            return exampleDic
        except Exception as e:
            print("+++++++=======================================")
            print(e)
            print(row)
            print("+++++++=======================================")

    data = pd.read_csv(file_path)
    data['id'] = data.apply(
    lambda x: getTheParaId(str(x['case_uri']), str(x['para_id'])), axis=1
    )
    examples =[]
    examples_dic = {}
    case_uris = data["case_uri"].unique()
    for test_case in case_uris:
        train_data = data[data["case_uri"]!= test_case]
        test_data = data[data["case_uri"] == test_case]
        #check how many positive exaple in each traing set
        #pick up the same number of negative
        #if total picked negatice examples are less than positive examples -- drop the positive-- keep the number equal
        # Separate positive and negative examples in the training data
        positive_examples = train_data[train_data['classifier_label'] == 1]
        negative_examples = train_data[train_data['classifier_label'] == 0]
        
        # Determine the minimum count between positive and negative examples
        min_count = min(len(positive_examples), len(negative_examples))
        
        # Balance the examples by picking 'min_count' from both positive and negative examples
        positive_examples = positive_examples.sample(n=min_count, random_state=42)
        negative_examples = negative_examples.sample(n=min_count, random_state=42)
        
        for i,row in positive_examples.iterrows():
            examples.append(getTheExampleTuple(row))
        for i,row in negative_examples.iterrows():
            examples.append(getTheExampleTuple(row))
        
        examples_dic[test_case]=examples
        examples = []
    return examples_dic

def get_optimal_batch_size(data, model_name, max_tokens_per_batch=80000, sample_size=100):
    """
    Determine optimal batch size based on text length and model constraints.
    
    Args:
        data: DataFrame containing paragraphs
        model_name: Name of the model being used
        max_tokens_per_batch: Maximum tokens to include in a batch
        sample_size: Number of random samples to evaluate
        
    Returns:
        int: Recommended batch size
    """
    # Get token encoder based on model type
    if "gpt" in model_name.lower():
        from tiktoken import encoding_for_model
        enc = encoding_for_model("gpt-4")
        
        # Sample random paragraphs and count tokens
        if len(data) > sample_size:
            sample = data.sample(sample_size)
        else:
            sample = data
            
        token_counts = [len(enc.encode(str(row['paragraphs']))) for _, row in sample.iterrows()]
        
    elif "claude" in model_name.lower():
        sample = data
        
        # Claude uses approximately 4 chars per token as a rough estimate
        token_counts = [len(str(row['paragraphs'])) // 4 for _, row in sample.iterrows()]
    else:
        sample = data
        # Default case for other models like LLama, assuming 5 chars per token as a rough estimate
        token_counts = [len(str(row['paragraphs'])) // 5 for _, row in sample.iterrows()]
    
    # Calculate statistics
    avg_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    
    # Add overhead for prompt template and format instructions (approx. 500 tokens)
    tokens_per_item = avg_tokens + 500
    
    # Calculate batch size based on average + buffer
    recommended_batch_size = max(1, int(max_tokens_per_batch / tokens_per_item))
    
    # Cap batch size based on rate limits (conservative value)
    #rate_limit_cap = 20 if "gpt" in model_name.lower() else 15
    
    return recommended_batch_size#min(recommended_batch_size, rate_limit_cap)



def process_csv_with_openai(example_json_file,caselaw_csv_to_process,model,batch_size=20,delay=0,max_tokens_per_batch=80000):
    def extract_phrases_or_error(x,column_name):
        try:
            return x[column_name]
        except KeyError:
            return "Error"
    def process_in_batches(data,parser,chain, batch_size=20,delay=0):
        responses = []
        # Split the DataFrame into batches
        print("batch size is",batch_size)
        
        for start in range(0, len(data), batch_size):
            print("starting again...")
            end = start + batch_size
            batch = data.iloc[start:end]
            
            # Prepare the input dictionary list for batch processing
            input_batch = [
                {
                    "para_id": row['para_id'],
                    "para_content": row['paragraphs'],
                    "format_instructions": parser.get_format_instructions()
                }
                for _, row in batch.iterrows()
            ]
            
            # Call chain.batch on the batch of inputs
            try:
                batch_responses = chain.batch(input_batch)
                #batch_responses = batch_responses.replace('```json', '').replace('```', '').strip()

            except Exception as e:
                
                print("Error in batch processing, processing individually")
                print(e)
                for item in input_batch:
                    try:
                        response = chain.invoke(item)
                        #response = response.replace('```json', '').replace('```', '').strip()
                        responses.append(response)
                    except Exception as e:
                        #response = util.getJsonList(response)
                        print("Error in individuall processing as well")
                        print(e)
                        if isinstance(response, str):
                            response = response.replace('```json', '').replace('```', '').strip()
                        responses.append(response)
                        # responses.append({"para_id":item['para_id'],"if_law_applied":"Error","application_of_law_phrases":"Error","reason":"Error"})
                continue
            responses.extend(batch_responses)
            time.sleep(delay) # to make sure it does not exceed the processing limit
            
        
        # Add responses to a new column in the original DataFrame
        return responses

    """
    Process a CSV file containing legal text using OpenAI handlers.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed results with interpretations
    """
    # Read the CSV file
    with open(example_json_file, 'r') as file:
        examples = json.load(file)
    
    # Initialize the chains
    parser,classifier_chain = getLegalClassifierChain(examples,model)  # Empty examples list for now
    
    df = pd.read_csv(caselaw_csv_to_process,index_col=False)
    # Usage in your code
    #optimal_batch_size = get_optimal_batch_size(df, model,max_tokens_per_batch)
    #responses = process_in_batches(df, parser, classifier_chain, batch_size=optimal_batch_size)
    Processeddata = process_in_batches(df,parser,classifier_chain,batch_size,delay)
    dataProcess = pd.DataFrame()
    dataProcess['gpt-40'] = Processeddata
    dataProcess['para_id'] = dataProcess['gpt-40'].apply(lambda x: x.get('para_id') if isinstance(x, dict) else None)
    dataProcess['if_law_applied'] = dataProcess['gpt-40'].apply(lambda x: x.get('if_law_applied') if isinstance(x, dict) else None)
    dataProcess['application_of_law_phrases'] = dataProcess['gpt-40'].apply(lambda x: x.get('application_of_law_phrases') if isinstance(x, dict) else None)
    dataProcess['reason_of_choosing_it_as_application'] = dataProcess['gpt-40'].apply(lambda x: x.get('reason') if isinstance(x, dict) else None)
    dataProcess = dataProcess.drop_duplicates(subset='para_id')
    df = df.merge(dataProcess,on="para_id")
    df['if_law_applied'] = df['if_law_applied'].apply(lambda x: '1' if str(x).lower() in ['1', 'true'] else '0')
    if 'gpt-40' in df.columns:
        df.drop(columns=['gpt-40'], inplace=True)
    df.to_csv(caselaw_csv_to_process,index=False)
    

if __name__ == "__main__":
    csv_path = "data/dataP.csv"  # replace with your actual CSV file path
    examples_dic = getExamples(csv_path)
    first_value = next(iter(examples_dic.values()))
    print(len(first_value))
    with open('data/examples.json', 'w') as json_file:
        json.dump(first_value, json_file, indent=4)
    
    #print(getExamples(csv_path))

