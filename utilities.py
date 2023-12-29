import re
import string
import nltk
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

