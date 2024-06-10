from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor, load_tools
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate


load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"
open_ai = OpenAI(temperature=0.0)
chat = ChatOpenAI(temperature=0.6, model=llm)
llm_math = LLMMathChain.from_llm(llm=open_ai)


# Manual way to define tools
#math_tool = Tool(
#    name="calculator",
#    func=llm_math.run,
#    description="Useful for when you need to answer questions related to math"
#) 

#tools = [math_tool]

# load_tools is bundles with langchain
tools = load_tools(['llm-math'], llm=open_ai)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

agent = create_tool_calling_agent(
    tools=tools,
    llm=chat,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=3, verbose=True)
answer = agent_executor.invoke({"input": "What color are polar bears?"})

print(f"Input: {answer['input']}")
print(f"Output: {answer['output']}")

