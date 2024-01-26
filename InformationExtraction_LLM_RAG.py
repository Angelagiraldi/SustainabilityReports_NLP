import os
import pandas as pd
from time import perf_counter
import yaml
from transformers import pipeline
from preprocessing_utils import *

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto





collection = UnstructuredCollector()

user_input = ["DataFactSheet-2022MicrosoftSustainabilityReport.pdf", "2022MicrosoftEnvironmentalSustainabilityReport.pdf"]

for input_file in user_input:
    print(f"Processing file: {input_file}")
    pdf_parser = ParsePDF(input_file)
    content = pdf_parser.extract_contents()
    print(f"Extracted content from PDF: {input_file}")
    collection.add_document(input_file, content)

# Display loaded documents
doc_titles = collection.list_doc_titles()
print(f"Loaded {len(doc_titles)} documents with the following titles: \n{doc_titles}")

# Example of accessing a document's raw text
print(collection.get_raw_text("2022MicrosoftEnvironmentalSustainabilityReport.pdf"))

# Load YAML configuration
with open('config.yaml', 'r') as file:
    yaml_config = yaml.safe_load(file)

# Define a chunk size
chunk_size = 1000

# Initialize the pre-trained model
#################################################################
# Tokenizer
#################################################################

model_name='mistralai/Mistral-7B-Instruct-v0.1'

model_config = transformers.AutoConfig.from_pretrained(
    model_name,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
inputs_not_chat = tokenizer.encode_plus("[INST] Tell me about fantasy football? [/INST]", return_tensors="pt")['input_ids'].to('cuda')

generated_ids = model.generate(inputs_not_chat, 
                               max_new_tokens=1000, 
                               do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)



# Process each document in collection
for title in collection.list_doc_titles():
    document = collection.get_raw_text(title)
    document_chunks = chunk_document(document, chunk_size=500)

    data = []
    for chunk in document_chunks:
        for column in yaml_config['columns']:
            prompt = column['prompt']
            response = model(chunk + '\n' + prompt)[0]['generated_text']
            # Extract and process the response as per your requirement
            data.append(response)

    # Convert the data into a structured format
    df = pd.DataFrame(data, columns=[col['name'] for col in yaml_config['columns']])
    print(f"Data for document {title}:")
    print(df)