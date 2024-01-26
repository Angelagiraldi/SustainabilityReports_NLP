import os
import pandas as pd
from time import perf_counter
import yaml
from transformers import pipeline
from preprocessing_utils import *


collection = UnstructuredCollector()

user_input = ["DataFactSheet-2022MicrosoftSustainabilityReport.pdf", "2022MicrosoftEnvironmentalSustainabilityReport.pdf"]

for input_file in user_input:
    print(f"Processing file: {input_file}")
    pdf_parser = ParsePDF(input_file)
    content = pdf_parser.extract_contents()
    print(f"Extracted content from PDF: {input_file}")
    collection.add_document(input_file, content)

# Display loaded documents
doc_titles = collection.list_doc_titles()
print(f"Loaded {len(doc_titles)} documents with the following titles: \n{doc_titles}")

# Example of accessing a document's raw text
print(collection.get_raw_text("2022MicrosoftEnvironmentalSustainabilityReport.pdf"))

# Load YAML configuration
with open('config.yaml', 'r') as file:
    yaml_config = yaml.safe_load(file)

# Define a chunk size
chunk_size = 1000

# Initialize the pre-trained model
model = pipeline('text-generation', model=yaml_config['model'])  # Replace with your model of choice

# Process each document in collection
for title in collection.list_doc_titles():
    document = collection.get_raw_text(title)
    document_chunks = chunk_document(document, chunk_size=500)

    data = []
    for chunk in document_chunks:
        for column in yaml_config['columns']:
            prompt = column['prompt']
            response = model(chunk + '\n' + prompt)[0]['generated_text']
            # Extract and process the response as per your requirement
            data.append(response)

    # Convert the data into a structured format
    df = pd.DataFrame(data, columns=[col['name'] for col in yaml_config['columns']])
    print(f"Data for document {title}:")
    print(df)