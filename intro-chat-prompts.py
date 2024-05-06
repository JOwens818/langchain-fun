import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"



customer_review = """
    Your product is terrible!  I don't know how you were able to get this 
    to the market.  I don't want this!  Actually no one should want this.  
    Seriously, give me my money back!
"""

template_string = f"""
    Rewrite the following {customer_review} in a polite tone, and then please
    translate the new review message into Portugese
"""


# Using OpenAI 
def get_completion(prompt, model=llm):
    client = OpenAI()
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


# Using langchain wrapper
def lanchain_prompt_template():
    model = ChatOpenAI(model=llm, temperature=0.7)
    prompt_template = ChatPromptTemplate.from_template(template_string)
    translate_message = prompt_template.format_messages(customer_review=customer_review)
    response = model(translate_message)
    return response.content

#rewrite = get_completion(template_string)
rewrite = lanchain_prompt_template()
print(rewrite)