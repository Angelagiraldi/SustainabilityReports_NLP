# Imports
from collections import defaultdict
import pandas as pd
import torch
from transformers import pipeline  # Hugging Face

from preprocessing_utils import *

pd.set_option("display.max_colwidth", None)


nestle_url = "https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/OTC_NSRGY_2022.pdf"
pdf_parser = ParsePDF(nestle_url)
content = pdf_parser.extract_contents()
sentences = pdf_parser.clean_text(content)


print(f"The Nestlè CSR report has {len(sentences):,d} sentences")

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


# Define and Create the zero-shot learning model
model_name = "microsoft/deberta-v2-xlarge-mnli" 
    # a smaller version: "microsoft/deberta-base-mnli"
ZSC = ZeroShotClassifier()
ZSC.create_zsl_model(model_name)
    # Note: the warning is expected, so ignore it


# Classify all the sentences in the report
    # Note: this takes a while
classified = ZSC.text_labels(sentences, esg_categories)
classified.sample(n=20)  # display 20 random records


# Look at an example of "E" classified sentences:
E_sentences = classified[classified.scores.gt(0.8) & classified.ESG.eq("E")].copy()
E_sentences.head(10)