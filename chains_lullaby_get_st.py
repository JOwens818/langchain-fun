from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
import streamlit as st

load_dotenv(find_dotenv())
llm = "gpt-3.5-turbo"
open_ai = OpenAI(temperature=0.7)


def generate_lullaby(location, name, language):
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

    response = overall_chain({"location": location, "name": name, "language": language})
    return response


def main():
    st.set_page_config(page_title="Generate Children's Lullaby", layout="centered")
    st.title("Let AI Write and Translate a Lullaby for You!")
    st.header("Get Started...")

    location_input = st.text_input(label="Where is the story set?")
    character_input = st.text_input(label="Where is the main character's name?")
    language_input = st.text_input(label="Translate this story into...")
    submit_button = st.button("Submit")

    if location_input and character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby..."):
                response = generate_lullaby(location_input, character_input, language_input)
            
                with st.expander("English Version"):
                    st.write(response['story'])

                with st.expander(f"{language_input} version"):
                    st.write(response['translated'])
            
            st.success("Lullaby Successfully Generated!")

if __name__ == '__main__':
    main()