from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.9, model=llm)
open_ai = OpenAI(temperature=0.0)

template = """
As a  children's book writer, please come up with a simple and short (90 words or less)
lullaby based on the location 
{location}
and the main character {name}

STORY:
"""

prompt = PromptTemplate(
    input_variables=["location", "name"],
    template=template
)

chain_story = LLMChain(llm=open_ai, prompt=prompt)
story = chain_story({"location": "New Jersey", "name": "Pooky"})
print(story['text'])
