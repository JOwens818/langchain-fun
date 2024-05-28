from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain


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

chain_story = LLMChain(llm=open_ai, prompt=prompt, output_key="story")


# === sequential chain ======
template_update = """
Translate the {story} into {language}.  Make sure the language is simple and fun.

TRANSLATION:
"""

prompt_translate = PromptTemplate(input_variables=["story", "language"], template=template_update)
chain_translate = LLMChain(llm=open_ai, prompt=prompt_translate, output_key="translated")

# Create sequential chain
overall_chain = SequentialChain(
    chains=[chain_story, chain_translate],
    input_variables=["location", "name", "language"],
    output_variables=["story", "translated"]
)

response = overall_chain({"location": "New Milford", "name": "Jimbo", "language": "Italian"})

print(f"English version: {response['story']}\n \n")
print(f"Translated version: {response['translated']}")