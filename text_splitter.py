from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

with open("/home/jbo/Downloads/MLK.txt") as paper:
    speech = paper.read()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function = len
)

texts = text_splitter.create_documents([speech])
print(texts)