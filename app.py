import os
import streamlit as st
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_huggingface_response(question):
    if question.strip() == "":
        return "Por favor, insira uma pergunta para obter uma resposta."

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.7, "max_length": 100},
        huggingfacehub_api_token=hf_token
    )
    response = llm(question)
    return response


st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

user_input = st.text_input("Input: ", key="input")
response = get_huggingface_response(user_input)

submit = st.button("Ask the question")

if submit:
    if user_input:
        try:
            response = get_huggingface_response(user_input)
            st.subheader("The response is:")
            st.write(response)
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
    else:
        st.error("Por favor, insira uma pergunta válida.")