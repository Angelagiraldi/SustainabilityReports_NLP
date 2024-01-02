# Imports
from collections import defaultdict
import pandas as pd
import torch
from transformers import pipeline  # Hugging Face

from preprocessing_utils import *

pd.set_option("display.max_colwidth", None)


nestle_url = "https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/NASDAQ_ABNB_2022.pdf"
pdf_parser = ParsePDF(nestle_url)
content = pdf_parser.extract_contents()
sentences = pdf_parser.clean_text(content)

print(f"The Nestlè CSR report has {len(sentences):,d} sentences")

print("Define categories")
# Define categories we want to classify
esg_categories = {
  "emissions": "E",
  "natural resources": "E",
  "pollution": "E",
  "diversity and inclusion": "S",
  "philanthropy": "S",
  "health and safety": "S",
  "training and education": "S",
  "transparancy": "G",
  "corporate compliance": "G",
  "board accountability": "G"}

print("Define model")
# Define and Create the zero-shot learning model
#model_name = "microsoft/deberta-v2-xlarge-mnli" 
model_name = "facebook/bart-large-mnli"
    # a smaller version: "microsoft/deberta-base-mnli"
print("Define zero shot classifier")
ZSC = ZeroShotClassifier()
print("Define zero shot model")
ZSC.create_zsl_model(model_name)
    # Note: the warning is expected, so ignore it

# Classify all the sentences in the report
    # Note: this takes a while
print("Classify")
classified = ZSC.text_labels(sentences, esg_categories)



# Check if the number of sentences matches the number of classified results
if len(sentences) == len(classified):
    classified['sentence'] = sentences
else:
    print(len(sentences))
    print(len(classified))
    print("Error: The number of sentences does not match the number of classified results.")
    # Optional: Handle the mismatch case, e.g., by aligning sentences with classifications

# Check if 'sentence' column has been successfully added before attempting to display
if 'sentence' in classified.columns:
    print("Available columns in classified DataFrame:", classified.columns)
    print("Display 20 random sentences with classifications")
    print(classified[['sentence', 'label', 'score', 'ESG']].sample(n=20))

    print("Display E sentences:")
    if 'score' in classified.columns and 'ESG' in classified.columns:
        E_sentences = classified[classified.score.gt(0.8) & classified.ESG.eq("E")][['sentence', 'label', 'score', 'ESG']].copy()
        print(E_sentences.head(10))
else:
    print("Sentence column not added. Displaying results without sentences.")
    print(classified[['label', 'score', 'ESG']].sample(n=20))