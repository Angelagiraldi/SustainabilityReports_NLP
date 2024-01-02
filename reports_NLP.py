# Imports
from collections import defaultdict
import pandas as pd
import torch
from transformers import pipeline  # Hugging Face
import os
os.environ['HF_HOME'] = '/afs/cern.ch/work/a/angirald/.cache/huggingface/hub'

from preprocessing_utils import *

pd.set_option("display.max_colwidth", None)


airbnb_url = "https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/NASDAQ_ABNB_2022.pdf"
cern_url = "https://cernbox.cern.ch/s/jjhyqEubYm9eIyF"
pdf_parser = ParsePDF(cern_url)
content = pdf_parser.extract_contents()
sentences = pdf_parser.clean_text(content)

print(f"The Airbnb CSR report has {len(sentences):,d} sentences")

print(">>>  Define categories")
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

print(">>>  Define model")
# Define and Create the zero-shot learning model
#model_name = "microsoft/deberta-v2-xlarge-mnli" 
model_name = "facebook/bart-large-mnli"
    # a smaller version: "microsoft/deberta-base-mnli"
print(">>>  Define zero shot classifier")
ZSC = ZeroShotClassifier()
print(">>>  Create zero shot model")
ZSC.create_zsl_model(model_name)
    # Note: the warning is expected, so ignore it

# Classify all the sentences in the report
    # Note: this takes a while
print(">>> Classify: ")
classified = ZSC.text_labels(sentences, esg_categories)


# Initialize a dictionary to hold aggregated results
aggregated_results = {sentence: {'labels': [], 'scores': [], 'ESG': []} for sentence in sentences}

# Iterate over the classified results and aggregate information for each sentence
for index, classification in classified.iterrows():
    sentence = sentences[index // len(esg_categories)]
    aggregated_results[sentence]['labels'].append(classification['label'])
    aggregated_results[sentence]['scores'].append(classification['score'])
    aggregated_results[sentence]['ESG'].append(classification['ESG'])

# Iterate over the aggregated results to normalize scores
for sentence_data in aggregated_results.values():
    total_score = sum(sentence_data['scores'])
    if total_score > 0:
        sentence_data['scores'] = [score / total_score for score in sentence_data['scores']]

# Convert aggregated results to a DataFrame
aggregated_df = pd.DataFrame([(sentence, data['labels'], data['scores'], data['ESG']) 
                              for sentence, data in aggregated_results.items()], 
                             columns=['sentence', 'labels', 'scores', 'ESG'])

# Display 20 random sentences with aggregated classifications
print(">>>  Display 20 random sentences with classifications")
print(aggregated_df.sample(n=20))

# Display sentences with high 'E' classification scores
print(">>>  Display E sentences:")
E_sentences = aggregated_df[aggregated_df['ESG'].apply(lambda esg_list: 'E' in esg_list) & 
                            aggregated_df['scores'].apply(lambda scores: any(score > 0.8 for score in scores))]
print(E_sentences.head(10))

# Invert the esg_categories mapping
category_to_labels = {}
for label, category in esg_categories.items():
    if category not in category_to_labels:
        category_to_labels[category] = []
    category_to_labels[category].append(label)

# Number of random samples per category
num_samples = 3
# Iterate over each category name
for category, labels in category_to_labels.items():
    print(f"\nCategory: {category}")
    
    # Filter sentences belonging to the current category
    category_sentences = aggregated_df[aggregated_df['labels'].apply(lambda x: any(label in x for label in labels))]
    
    # Sample sentences
    sampled_sentences = category_sentences.sample(min(num_samples, len(category_sentences)))

    # Display sampled sentences with their scores
    for _, row in sampled_sentences.iterrows():
        sentence = row['sentence']
        scores = row['scores']
        category_labels = row['labels']
        # Find the scores corresponding to the current category
        category_scores = [scores[i] for i, label in enumerate(category_labels) if label in labels]
        print(f"Sentence: {sentence}\nScores: {category_scores}\n")


file_path = "aggregated_results.csv"  # You can change this to your desired file path
# Save the DataFrame to a CSV file
aggregated_df.to_csv(file_path, index=False, encoding='utf-8-sig')
print(f">>> Aggregated results saved to {file_path}")