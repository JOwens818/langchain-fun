from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.0, model=llm)
embeddings = OpenAIEmbeddings()

text1 = "Math is a great subject to study"
text2 = "Dogs are friendly when they are happy and well fed"
text3 = "Physics is not one of my favorite subjects"

embed1 = embeddings.embed_query(text1)
embed2 = embeddings.embed_query(text2)
embed3 = embeddings.embed_query(text3)

#print(embed1)

similarity = np.dot(embed1, embed2)
print(f"Similarity %: {similarity * 100}")