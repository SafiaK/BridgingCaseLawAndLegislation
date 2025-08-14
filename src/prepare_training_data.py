import pandas as pd
import json

from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv('data_for_training.csv')

# Group by the input columns and aggregate key phrases into a list
grouped = df.groupby(['url', 'para_id', 'paragraphs', 'section_text','case_term','legislation_term'], as_index=False).agg({
    'key_phrases': lambda x: list(x)
})

# Define the system instruction
system_prompt = (
    "You are given a case law paragraph and a legislation section text from a UK act. "
    "Your task is to extract excerpt from the case law which are legally interpreted as the case_law_excerpt, "
    "the corresponding legislation excerpt from the section text as legislation_excerpt, "
    "and then extract noun phrases from the legislation text as legislation_key_phrases."
)
print(grouped.head())
grouped = grouped.rename(columns={
    'case_term': 'case_law_excerpt',
    'legislation_term': 'legislation_excerpt',
    'key_phrases': 'legislation_key_phrases'
})
# Group by the input columns and aggregate the results into a list of dictionaries
groupedb = grouped.groupby(['url', 'para_id', 'paragraphs', 'section_text'], as_index=False).apply(
    lambda x: pd.Series({
        'result': x.apply(lambda row: {
            'case_law_excü