import pandas as pd

# Read the CSV file
df = pd.read_csv('data/filipino.csv')
df['relevant_answers'] = df['relevant_answers'].apply(lambda x: x.split(';'))
df['irrelevant_answers'] = df['irrelevant_answers'].apply(lambda x: x.split(';'))
df['prompt_variation_id'] = df['prompt_variation_id'].astype(str)
df['prompt_id'] = df['prompt_variation_id'].astype(str)

# Convert to Parquet format
df.to_parquet('data/filipino.parquet', engine='pyarrow')