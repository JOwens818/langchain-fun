from dotenv import find_dotenv, load_dotenv
from langchain_ibm import WatsonxLLM
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import os

load_dotenv(find_dotenv())
llm = "ibm/granite-13b-instruct-v2"
chat = WatsonxLLM(
    model_id=llm,
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.environ["WATSONX_PROJECT_ID"],
    apikey=os.environ["WATSONX_APIKEY"],
    params={
        "decoding_method": "sample",
        "max_new_tokens": 100,
        "min_new_tokens": 1,
        "temperature": 0.0,
        "top_k": 50,
        "top_p": 1,
    }
)


# load_tools is bundles with langchain
tools = load_tools(['llm-math'], llm=chat)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

memory = ConversationBufferMemory(
  memory_key="chat_history"
)

agent = create_tool_calling_agent(
    tools=tools,
    llm=chat,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=3, verbose=True, memory=memory)


while True:
    print("===========================================================")
    user_input = input("Me: ")
    if user_input.lower() == "exit":
        break

    response = agent_executor.invoke({
        "input": user_input
    })

    print("AI: ", response["output"])
    print("===========================================================")

