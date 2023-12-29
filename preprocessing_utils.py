import pandas as pd
import re
import string
import nltk
from transformers import pipeline
from tika import parser

class ParsePDF:
    """
    A class to parse and process text from a PDF file.

    Attributes:
        url (str): The URL or path to the PDF file.

    Methods:
        extract_contents(): Extracts text from the PDF file.
        clean_text(text): Cleans and tokenizes the extracted text into sentences.
        remove_non_ascii(text): Removes non-ASCII characters from the text.
        replace_tabs_with_spaces(text): Replaces tabs with spaces in the text.
        aggregate_lines(text): Aggregates lines based on certain conditions.
        tokenize_sentences(text): Tokenizes the text into sentences using NLTK.
        clean_sentence(sentence): Performs additional cleaning on each sentence.
    """

    def __init__(self, url):
        """
        Initializes the ParsePDF object with the URL or path of the PDF file.

        Args:
            url (str): The URL or path to the PDF file.
        """
        self.url = url

    def extract_contents(self):
        """
        Extracts text from the PDF file using the Tika parser.

        Returns:
            str: The extracted text content from the PDF.
        """
        try:
            pdf = parser.from_file(self.url)
            return pdf.get("content", "")
        except Exception as e:
            print(f"Error extracting PDF contents: {e}")
            return ""

    def clean_text(self, text):
        """
        Cleans and tokenizes the extracted text into sentences.

        Args:
            text (str): The raw text extracted from the PDF.

        Returns:
            list: A list of cleaned and tokenized sentences.
        """
        text = self.remove_non_ascii(text)
        text = self.replace_tabs_with_spaces(text)
        lines = self.aggregate_lines(text)
        return self.tokenize_sentences(" ".join(lines))

    def remove_non_ascii(self, text):
        """
        Removes non-ASCII characters from the text.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text with only ASCII characters.
        """
        return "".join(filter(lambda x: x in set(string.printable), text))

    def replace_tabs_with_spaces(self, text):
        """
        Replaces tabs with spaces in the text.

        Args:
            text (str): The text with tabs.

        Returns:
            str: The text with tabs replaced by spaces.
        """
        return re.sub(r"\t+", " ", text)

    def aggregate_lines(self, text):
        """
        Aggregates lines based on certain conditions. Lines that start with a space,
        are lowercase, or do not end with a period are combined with the previous line.
        Uppercase lines are treated as headers and skipped.

        Args:
            text (str): The text to be aggregated.

        Returns:
            list: A list of aggregated lines.
        """
        fragments = []
        prev = ""
        for line in re.split(r"\n+", text):
            if line.isupper():
                prev = "."  # Treat as a header
            elif line and (line.startswith(" ") or line[0].islower() or not prev.endswith(".")):
                prev = f"{prev} {line}".strip()
            else:
                fragments.append(prev.strip())
                prev = line
        fragments.append(prev.strip())
        return fragments

    def tokenize_sentences(self, text):
        """
        Tokenizes the text into sentences using NLTK.

        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of tokenized sentences.
        """
        sentences = []
        for sentence in nltk.sent_tokenize(text):
            cleaned_sentence = self.clean_sentence(sentence)
            if "table of contents" not in cleaned_sentence and len(cleaned_sentence) > 5:
                sentences.append(cleaned_sentence)
        return sentences

    def clean_sentence(self, sentence):
        """
        Performs additional cleaning on each sentence, such as converting to lowercase.

        Args:
            sentence (str): The sentence to be cleaned.

        Returns:
            str: The cleaned sentence.
        """
        sentence = sentence.lower()  # Convert to lowercase
        # Add more cleaning steps as needed
        return sentence



from transformers import pipeline
import pandas as pd

class ZeroShotClassifier:
    """
    A class for performing zero-shot classification using Hugging Face's transformers.

    Methods:
        create_zsl_model(model_name): Initializes the zero-shot learning model.
        classify_text(text, categories, multi_label): Classifies text into predefined categories.
        text_labels(text, category_dict, cutoff): Classifies text and formats the results, applying an optional cutoff.
    """

    def __init__(self):
        """
        Initializes the ZeroShotClassifier without a model.
        The model needs to be created using create_zsl_model method.
        """
        self.model = None

    def create_zsl_model(self, model_name):
        """
        Initializes the zero-shot learning model with the specified model.

        Args:
            model_name (str): The name of the model to be used for classification.

        Returns:
            bool: True if the model is successfully created, False otherwise.
        """
        try:
            self.model = pipeline("zero-shot-classification", model=model_name)
            return True
        except PipelineException as e:
            print(f"Error initializing the model: {e}")
            return False
    
    def classify_text(self, text, categories, multi_label=True):
        """
        Classifies text(s) into predefined categories using the zero-shot classification model.

        Args:
            text (str or list): The text or list of texts to classify.
            categories (list): A list of categories for classification.
            multi_label (bool): Whether multiple labels can be assigned to each text.

        Returns:
            dict: The classification results containing labels and scores.
        """
        if not self.model:
            raise ValueError("Model not initialized. Call create_zsl_model first.")

        hypothesis_template = "This text is about {}."
        try:
            result = self.model(text, categories, multi_label=multi_label, hypothesis_template=hypothesis_template)
            return result
        except Exception as e:
            print(f"Error during classification: {e}")
            return {}

    def text_labels(self, text, category_dict, cutoff=None):
        """
        Classifies text into predefined categories and formats the results. 
        Optionally applies a score cutoff to filter results.

        Args:
            text (str): The text to classify.
            category_dict (dict): A dictionary mapping categories to labels.
            cutoff (float, optional): A score threshold for filtering results.

        Returns:
            DataFrame: A pandas DataFrame with classification results.
        """
    categories = list(category_dict.keys())
    result = self.classify_text(text, categories, multi_label=True)

    if not result:
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

    # Process each label and score, applying the cutoff if provided
    processed_results = []
    for label, score in zip(result["labels"], result["scores"]):
        if cutoff is None or score > cutoff:
            processed_results.append({
                "label": label,
                "score": score,
                "ESG": category_dict.get(label, "Unknown")
            })

    return pd.DataFrame(processed_results)