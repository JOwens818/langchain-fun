from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain 
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.6, model=llm)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=chat, memory=memory, verbose=True)

conversation("Hello there, my name is Jim")
conversation("Why is the sky blue?")
conversation("How long has the New York Yankees been a baseball team?")
conversation("What is my name?")

print(conversation)