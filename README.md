# Sustainability Reports Analysis Using NLP and Zero-Shot Learning


This project focuses on leveraging Natural Language Processing (NLP) and Zero-Shot Learning (ZSL) to analyze and classify sustainability reports. These reports, crucial in understanding a company's commitment to Environmental, Social, and Governance (ESG) principles, offer insights into corporate responsibility and ethical practices. 
My goal is to provide a Python-based tool that parses PDF sustainability reports and categorizes content using advanced machine learning techniques.

## What is ESG?
ESG stands for Environmental, Social, and Governance - three critical factors in assessing the ethical impact and sustainability practices of a company.

* **Environmental**: This aspect considers how a company performs as a steward of nature. It includes climate change policies, waste management, and energy efficiency.
* **Social**: This area examines how the company manages relationships with employees, suppliers, customers, and communities. It includes workplace diversity, human rights, and consumer protection.
* **Governance**: This pertains to a company’s leadership, executive pay, audits, internal controls, and shareholder rights.

## Importance of ESG Analysis
ESG analysis is becoming increasingly important in the business world. Companies with high ESG scores are often associated with higher profitability, lower volatility, and better capital allocation. These companies tend to have higher valuations and returns, making them attractive to investors and stakeholders.

## Corporate Social Responsibility (CSR) Reports
CSR reports are documents published by companies to showcase their efforts in ESG areas. These reports, varying in length from 30 to over 200 pages, provide insights into a company's environmental and social impacts. While not standardized, they are a key resource for understanding a company's commitment to CSR.

## Natural Language Processing (NLP)
NLP is a field at the intersection of computer science, artificial intelligence, and linguistics. It involves programming computers to process and analyze large amounts of natural language data. In this project, NLP is used to interpret and understand the unstructured text in CSR reports.

## Zero-Shot Learning (ZSL)
ZSL is a machine learning technique where the model is trained to identify data that it has not seen during its training phase. It's particularly useful in classifying text into specific categories without needing extensive labeled datasets. In this context, ZSL helps categorize sentences in sustainability reports into relevant ESG categories, even if those categories were not part of the model's initial training data.

## Using NLP and ZSL for ESG Analysis
This Python tool uses NLP to process the text from CSR reports, breaking it down into manageable segments. Then, applying ZSL, it classifies these segments into predefined ESG categories. This method allows for a quick, efficient, and comprehensive analysis of lengthy reports, providing valuable insights into a company's sustainability practices.

## Conclusion
This project harnesses the power of NLP and ZSL to offer a novel approach to analyzing sustainability reports. By automating the classification of ESG-related content, I provide a valuable tool for investors, researchers, and anyone interested in corporate responsibility and ethical business practices.

#### Sources and References
* [ESG Matters - Harvard Corporate Governance](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/)
* [ESG Matters II - Harvard Corporate Governance](https://corpgov.law.harvard.edu/2021/06/02/esg-matters-ii/)
* [BlackRock ESG Investment Statement](https://www.blackrock.com/corporate/literature/publication/blk-esg-investment-statement-web.pdf)
* [Sustainability Reporting - GA Institute](https://www.ga-institute.com/index.php?id=9128)
* [Project Repository on GitHub](https://github.com/hannahawalsh/HTTF4-ESG-and-NLP)



***

## Parsing CSR PDFs

Parsing CSR reports is a critical step of the analysis. These reports are typically published as PDFs, a format that poses significant challenges for text extraction due to its focus on layout preservation. Our objective is to convert these reports into a computer-readable format, specifically extracting text as a list of sentences.

### Methodology
I decided to use a straightforward yet effective approach involving several tools:
* #### Apache Tika: 
A powerful content analysis toolkit, Tika helps us extract text from the PDFs.
* #### Regular Expressions:
I employ regular expressions to filter and join the text, ensuring that only relevant information is retained.
* #### Natural Language Toolkit (NLTK): 
NLTK is used for splitting the extracted text into sentences, a crucial step for further analysis.
While this method is not the most sophisticated, it's relatively simple and sufficiently effective for this initial purposes.

### Task-Specific Considerations
Text cleaning and parsing are highly task-specific. The adequacy of our approach depends on the specific requirements of the ESG analysis. Users should consider the nature of their reports and what's sufficient for their problem.

### Enhancements and Alternatives
While this current method serves the basic needs, I acknowledge its limitations. For more complex layouts or higher accuracy needs, alternative tools like PyPDF2 or PDFMiner might be more appropriate. Users are encouraged to explore these options if they encounter limitations with the current setup.

### Error Handling and Quality Assurance
Extracting text from PDFs can introduce errors, such as misinterpretation of characters or formatting issues. I recommend implementing the following strategies:
* Random Checks: Periodically compare extracted text with the original PDF to ensure accuracy.
* Character Encoding Checks: Ensure that the text is correctly encoded and special characters are handled appropriately.
* Error Logging: Implement logging to capture and review any anomalies during the extraction process.
* Performance Considerations
The efficiency of text extraction can vary based on the complexity and size of the PDFs. Monitor the processing time and resource usage, and consider batch processing or parallel execution for large sets of reports.

This approach to parsing CSR PDFs is designed to be a starting point, balancing simplicity and effectiveness. I encourage users to adapt and enhance this method based on their specific needs and challenges. By continuously refining the process, I aim to provide a robust foundation for ESG analysis using NLP and Zero-Shot Learning.