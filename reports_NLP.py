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


print(f"The McDonalds CSR report has {len(sentences):,d} sentences")
print(sentences[2])
print(sentences[1134])
print(sentences[12456])