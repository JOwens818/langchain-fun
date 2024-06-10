from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor, load_tools
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"
open_ai = OpenAI(temperature=0.0)
chat = ChatOpenAI(temperature=0.6, model=llm)


# load_tools is bundles with langchain
tools = load_tools(['llm-math'], llm=chat)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

    try:
        response = agent_executor.invoke({
            "input": user_input
        })
    except Exception as error:
        print(error)

    print("AI: ", response["output"])
    print("===========================================================")

