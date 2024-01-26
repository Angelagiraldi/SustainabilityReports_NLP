import os
import pandas as pd
from time import perf_counter
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
print(collection.get_raw_text("DataFactSheet-2022MicrosoftSustainabilityReport.pdf"))