from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain


load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.0, model=llm)


biology_template = """
You are a very smart biology professor.  You are great at answering questions about biology
in a concise and easy to understand manner.  When you don't know the answer to a question you
admit you do not know.

Here is the question:
{input}
"""

math_template = """
You are a very good mathematician.  You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts, 
answer the component parts, and then put them together to answer the broader question.

Here is the question:
{input}
"""

astronomy_template = """
You are a very good astronomer.  You are great at answering astronomy questions.  You are so 
good because you are able to break down hard problems into their component parts, answer the 
component parts, and then put them together to answer the broader question.

Here is the question:
{input}
"""

travel_agent_template = """ 
You are a very good travel agent with a large amount of knowledge when it comes to getting people 
the best deals and recommendations for travel, vacations, flights, and world's best destinations 
for vacation.  You are so good because you are able to break down hard problems into their 
component parts, answer the component parts, and then put them together to answer the broader question.

Here is the question:
{input}
"""


prompt_infos = [
    {
        "name": "biology",
        "description": "Good for answering biology related questions",
        "prompt_template": biology_template
    },
    {
        "name": "math",
        "description": "Good for answering math related questions",
        "prompt_template": math_template
    },    
    {
        "name": "astronomy",
        "description": "Good for answering astronomy related questions",
        "prompt_template": astronomy_template
    },    
    {
        "name": "travel_agent",
        "description": "Good for answering travel, tourism, and vacation questions",
        "prompt_template": travel_agent_template
    }
]

destination_chains = {}

for info in prompt_infos:
    name = info["name"]
    prompt_template = info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=chat, prompt=prompt)
    destination_chains[name] = chain


default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=chat, prompt=default_prompt)

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=['input'],
    output_parser=RouterOutputParser()
)

router_chain = LLMRouterChain.from_llm(
    llm=chat,
    prompt=router_prompt
)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

response = chain.run("What is the NY Yankees record as of today?")
print(response)