from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


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

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#answer = retriever.get_relevant_documents("Tell me how much my bill is")
#print(retriever.search_type)
#print(answer[0].page_content)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

while True:
    print("===========================================================")
    user_input = input("Me: ")
    if user_input.lower() == "exit":
        break

    response = chain.invoke({
        "input": user_input
    })

    print("AI: ", response["answer"])
    print("===========================================================")