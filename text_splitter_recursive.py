from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

with open("/home/jbo/Downloads/MLK.txt") as paper:
    speech = paper.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=100,
    length_function = len,
    add_start_index=True
)

docs = text_splitter.create_documents([speech])
print(docs[0])