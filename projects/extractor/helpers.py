from dotenv import find_dotenv, load_dotenv
import os
from pypdf import PdfReader
import pandas as pd
import re
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv(find_dotenv())

# Extract info from PDF file
def get_pdf_text(pdf_doc):
    text=""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Extract data from text
def extract_data(pages_data):
    template = """
        Extract all of the following values: Invoice ID, DESCRIPTION, Issue Date, 
        UNIT PRICE, AMOUNT, Bill For, From, and Terms from {pages}.  Remove any dollar symbols from output.

        Expected output: {{'Invoice ID': '1001234', 'DESCRIPTION': 'description', 'Issue Date': '5/4/2023', 'UNIT PRICE', '2', 'AMOUNT': '1100.00', 'Bill For': 'James', 'From': 'excel company', 'Terms': 'pay this now'}}
    """

    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    llm = OpenAI(temperature=0.7)
    full_response = llm(prompt_template.format(pages=pages_data))
    full_response = full_response.replace("$","")
    return full_response


# Create docs from the uploaded PDFs
def create_docs(user_pdf_list):
    df = pd.DataFrame({
        'Invoice ID': pd.Series(dtype='str'),
        'DESCRIPTION': pd.Series(dtype='str'),
        'Issue Date': pd.Series(dtype='str'),
        'UNIT PRICE': pd.Series(dtype='float'),
        'AMOUNT': pd.Series(dtype='float'),
        'Bill For': pd.Series(dtype='str'),
        'From': pd.Series(dtype='str'),
        'Terms': pd.Series(dtype='str'),
    })

    for filename in user_pdf_list:
        print(filename)
        raw_data = get_pdf_text(filename)
        llm_extracted_data = extract_data(raw_data)
        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            data_dict = eval('{' + extracted_text + '}')
            print(data_dict)
        else:
            print("No match found")

        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

        print("============================DONE==============================")
    
    df.head()
    return df