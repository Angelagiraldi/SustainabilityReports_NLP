import pandas as pd
import spacy
from preprocessing_utils import *


# Load Spacy model globally
nlp = spacy.load('en_core_web_sm')

token_dict = {
        'Org': 'G', 
        'Total emissions (Scope 1 + 2 + 3)': 'E', 
        'Category 6 – Business Travel': 'E',
        'GHG emissions within carbon neutral boundary': 'E', 
        'Total renewable electricity consumption': 'E',
        'Total water discharges': 'E', 
        'Percentage of product packaging recyclability': 'E'
    }

def classify_text(self, text, categories, multi_label=True):
    if not self.model:
        raise ValueError("Model not initialized. Call create_zsl_model first.")

    hypothesis_template = "This text is about {}."
    try:
        result = self.model(text, categories, multi_label=multi_label, hypothesis_template=hypothesis_template)
        
        # Adjust the handling based on the type of 'text'
        if isinstance(text, str):  # Single text
            return [result]  # Wrap the result in a list for consistent handling
        else:  # Multiple texts
            return result
    except Exception as e:
        print(f"Error during classification: {e}")
        return []

def classify_entities_with_context(sentences, classifier, category_dict):
    """
    Classifies each sentence in the provided text into predefined categories.

    Args:
        sentences (list): A list of sentences to analyze.
        classifier (ZeroShotClassifier): An instance of the ZeroShotClassifier.
        category_dict (dict): Dictionary mapping categories to labels.

    Returns:
        dict: Dictionary with categorized sentences.
    """
    categorized_sentences = {category: [] for category in category_dict}

    for sentence in sentences:
        classification_result = classifier.text_structured_labels(sentence, category_dict)
        for result in classification_result:
            for label in result.get('labels', []):
                if label in category_dict and result.get('score', 0) > 0.6:  
                    categorized_sentences[label].append(sentence)
    return categorized_sentences

# User input for RSS link or PDF file
user_input = "DataFactSheet-2022MicrosoftSustainabilityReport.pdf"
print(f"Processing file: {user_input}")

pdf_parser = ParsePDF(user_input)
content = pdf_parser.extract_contents()
print("Extracted content from PDF.")

sentences = pdf_parser.clean_text(content)
print("Cleaned text from PDF.")

zsl_classifier = ZeroShotClassifier()
print("Initializing ZeroShotClassifier model...")
model_created = zsl_classifier.create_zsl_model('facebook/bart-large-mnli')

if model_created:
    print("Model created successfully. Classifying entities...")
    entity_info = classify_entities_with_context(sentences, zsl_classifier, token_dict)
    print("Classification completed.")
    print(entity_info)
else:
    print("Failed to create the ZeroShotClassifier model.")
