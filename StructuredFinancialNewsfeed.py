import pandas as pd
import spacy
import yfinance as yf
from preprocessing_utils import *

# Load Spacy model globally
nlp = spacy.load('en_core_web_sm')

token_dict = {
        'Org': [], 'Total emissions (Scope 1 + 2 + 3)': [], 'Category 6 – Business Travel': [],
        'GHG emissions within carbon neutral boundary': [], 'Total renewable electricity consumption': [],
        'Total water discharges': [], 'Percentage of product packaging recyclability': []
    }


def classify_entities_with_context(text, stocks_df, classifier, category_dict = token_dict):
    """
    Extracts information based on the text and classifies sentences 
    into predefined categories for each entity.

    Args:
        text (str): The text to analyze.
        stocks_df (DataFrame): DataFrame containing stock information.
        classifier (ZeroShotClassifier): An instance of the ZeroShotClassifier.
        category_dict (dict): Dictionary mapping categories to labels.

    Returns:
        dict: Dictionary with entities and their associated sentences.
    """
    
    doc = nlp(text)
    entity_sentences = {}

    for token in doc.ents:
        if stocks_df['Company Name'].str.contains(token.text).sum():
            # Find sentences containing the entity
            sentences = [sent.text for sent in doc.sents if token.text in sent.text]
            entity_sentences[token.text] = {
                'sentences': [],
                'categories': []
            }

            # Classify each sentence and store relevant ones
            for sentence in sentences:
                classification_result = classifier.text_labels(sentence, category_dict)
                relevant_sentences = classification_result.loc[classification_result['score'] > 0.5]  # Adjust score threshold as needed
                if not relevant_sentences.empty:
                    entity_sentences[token.text]['sentences'].append(sentence)
                    entity_sentences[token.text]['categories'].extend(relevant_sentences['ESG'].unique().tolist())

    return entity_sentences

# Load stock data once at the beginning
stocks_df = pd.read_csv("./data/ind_nifty500list.csv")

# RSS link input
# User input for RSS link or PDF file
user_input = "DataFactSheet-2022MicrosoftSustainabilityReport.pdf"
pdf_parser = ParsePDF(user_input)
content = pdf_parser.extract_contents()
sentences = pdf_parser.clean_text(content)

zsl_classifier = ZeroShotClassifier()
model_created = zsl_classifier.create_zsl_model('facebook/bart-large-mnli')  


# Process the input source and display results
If model_created:
    entity_info = classify_entities_with_context(sentences, stocks_df, zsl_classifier, token_dict)
    print(entity_info)
else:
    print("Failed to create the ZeroShotClassifier model.")

