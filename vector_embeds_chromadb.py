from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.9, model=llm)
embeddings = OpenAIEmbeddings()

# 1. Load PDF
loader = PyPDFLoader("/home/jbo/Downloads/spectrum-mar-24.pdf")
docs = loader.load()

# 2. Split doc into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

print(vectorstore._collection.count())
query = "How much is my bill this month?"
resp = vectorstore.similarity_search(query=query, k=3)
print(len(resp))
print(resp[0].page_content)