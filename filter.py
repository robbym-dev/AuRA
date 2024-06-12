import pandas as pd

# Load the data from the TSV file
input_file_path = '/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_ratio_0.6_filtered.tsv'
data = pd.read_csv(input_file_path, sep='\t')

# Extract the 'Document' column
documents = data[['Query']].drop_duplicates()  # Drop duplicates if needed

# Save the extracted data to a new TSV file
output_file_path = '/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_queries.tsv'
documents.to_csv(output_file_path, sep='\t', index=False)