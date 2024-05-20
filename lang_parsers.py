from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.0, model=llm)

email_response = """
Here's our itinerary for our upcoming trip to Europe.
We leave from Denver, Colorado airport at 8:45 pm, and arrive in Amsterdam 10 hours later
at Shipol Airport.
We'll grab a ride to our airbnb and maybe stop somewhere for breakfast before taking a nap.

Some sightseeing will follow for a couple of hours.
We will then go shop for gifts
to bring back to our children and friends.

The next morning, at 7:45am we'll drive to Belgium, Brussels - it should only take a few hours.
While in Brussels we want to explore the city to its fullest - no rock left unturned.
"""

email_template = """
From the following email, extract the following information:

leave_time: when are they leaving for vacation to Europe.  If there's an actual
time written, use it, if not write unknown.

leave_from: where are they leaving from, the airport or city name and state if available.

cities_to_visit: extract the cities they are going to visit.  If there are more than 
one, put them in square brackets like '["cityone", "citytwo"]'.

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

email: {email}
"""

prompt_template = ChatPromptTemplate.from_template(email_template)
#print(prompt_template)

messages = prompt_template.format_messages(email=email_response)
response = chat(messages=messages)

#print(response.content)


#============== Langchain Parsers =============
leave_time_schema = ResponseSchema(
    name="leave_time",
    description="When they are leaving.  It's usually \
                 a numeric time of the day.  If not \
                 available write n/a"
)

leave_from_schema = ResponseSchema(
    name="leave_from",
    description="Where they are leaving from.  \
                 It's a city, airport, state, or province"
)

cities_to_visit_schema = ResponseSchema(
    name="cities_to_visit",
    description="The cities, towns they will be visiting on \
                 their trip.  This needs to be in a list"
)

response_schema = [
    leave_time_schema,
    leave_from_schema,
    cities_to_visit_schema
]

# setup output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()
#print(format_instructions)

revised_email_template = """
From the following email, extract the following information:

leave_time: when are they leaving for vacation to Europe.  If there's an actual
time written, use it, if not write unknown.

leave_from: where are they leaving from, the airport or city name and state if available.

cities_to_visit: extract the cities they are going to visit.  If there are more than 
one, put them in square brackets like '["cityone", "citytwo"]'.

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

email: {email}
{format_instructions}
"""

updated_prompt = ChatPromptTemplate.from_template(template=revised_email_template)
messages = prompt_template.format_messages(email=email_response, format_instructions=format_instructions)
response = chat(messages)
print(type(response.content))
print(response.content)
output_dict = output_parser.parse(response.content)
print(type(output_dict))
print(output_dict)