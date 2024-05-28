from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.9, model=llm)

loader = PyPDFLoader("/home/jbo/Downloads/spectrum-mar-24.pdf")
pages = loader.load()

print(len(pages))
print("===============================================")
page = pages[0]
print(page.page_content[0:700]) 
print("===============================================")
print(page.metadata)