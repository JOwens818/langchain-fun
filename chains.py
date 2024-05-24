from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.9, model=llm)
open_ai = OpenAI(temperature=0.0)

# LLMChain
prompt = PromptTemplate(
    input_variables=['language'],
    template="How do you say good morning in {language}"
)

chain = LLMChain(llm=open_ai, prompt=prompt)
print(chain.run(language="French"))