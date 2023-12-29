# Imports
from collections import defaultdict
import pandas as pd
import torch
from transformers import pipeline  # Hugging Face

from preprocessing_utils import *

pd.set_option("display.max_colwidth", None)


nestle_url = "https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/OTC_NSRGY_2022.pdf"
pp = utilities.ParsePDF(nestle_url)
pp.extract_contents()
sentences = pp.clean_text()


print(f"The McDonalds CSR report has {len(sentences):,d} sentences")