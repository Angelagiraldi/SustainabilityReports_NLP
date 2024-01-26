import pandas as pd
import requests
from bs4 import BeautifulSoup
import spacy
from spacypdfreader.spacypdfreader import pdf_reader

import streamlit as st
import yfinance as yf

# Load Spacy model globally to avoid reloading it in every function
nlp = spacy.load('en_core_web_sm')

# Set up Streamlit title
st.title('Buzzing Stocks :zap:')

def process_rss(headings):
    """
    Processes RSS feed headings for NLP analysis.
    """
    for title in headings:
        doc = nlp(title.text)
        # Process with NLP here (e.g., visualize NER)
        # Removed code for brevity
        print("Done----------")

def extract_text(input_source):
    """
    Extracts text from the given input source, which can be either
    an RSS link or a PDF file.
    """
    if input_source.lower().endswith('.pdf'):
        # Handle PDF file
        doc = pdf_reader(input_source, nlp)
        return [doc.text]  # Return a list containing the entire text of the PDF
    else:
        # Handle RSS link
        response = requests.get(input_source)
        soup = BeautifulSoup(response.content, features='lxml')
        return [title.text for title in soup.findAll('title')]

def stock_info(headings, stocks_df):
    """
    Extracts stock information based on the headings.
    """
    token_dict = {
        'Org': [], 'Symbol': [], 'currentPrice': [],
        'dayHigh': [], 'dayLow': [], 'forwardPE': [], 'dividendYield': []
    }
    for title in headings:
        doc = nlp(title.text)
        for token in doc.ents:
            # Check and retrieve stock information
            try:
                if stocks_df['Company Name'].str.contains(token.text).sum():
                    symbol = stocks_df[stocks_df['Company Name'].\
                                        str.contains(token.text)]['Symbol'].values[0]
                    org_name = stocks_df[stocks_df['Company Name'].\
                                        str.contains(token.text)]['Company Name'].values[0]
                    token_dict['Org'].append(org_name)
                    print(symbol+".NS")
                    token_dict['Symbol'].append(symbol)
                    stock_info = yf.Ticker(symbol+".NS").info
                    token_dict['currentPrice'].append(stock_info['currentPrice'])
                    token_dict['dayHigh'].append(stock_info['dayHigh'])
                    token_dict['dayLow'].append(stock_info['dayLow'])
                    token_dict['forwardPE'].append(stock_info['forwardPE'])
                    token_dict['dividendYield'].append(stock_info['dividendYield'])
                else:
                    pass
            except:
                pass
    return pd.DataFrame(token_dict)

# Load stock data once at the beginning
stocks_df = pd.read_csv("./data/ind_nifty500list.csv")

# RSS link input
# User input for RSS link or PDF file
user_input = st.text_input("Add your RSS link or upload a PDF file:", 
                           "https://www.moneycontrol.com/rss/buzzingstocks.xml")

# Determine if the input is a link or a file path (for PDF)
if user_input.startswith('http://') or user_input.startswith('https://'):
    # Process RSS feed
    fin_headings = extract_text(user_input)
else:
    # Assume it's a file path for a PDF
    fin_headings = extract_text(user_input)

# Process RSS feed and display results
fin_headings = extract_text_from_rss(user_input)
output_df = stock_info(fin_headings, stocks_df)
output_df.drop_duplicates(inplace=True)
st.dataframe(output_df)

# Display financial news in an expander
with st.expander("Expand for Financial News!"):
    for h in fin_headings:
        st.markdown("* " + h.text)
