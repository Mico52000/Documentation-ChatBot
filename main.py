import streamlit as st
from backend.core import run_llm
from streamlit_chat import message


def create_sources_string(source_urls):
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


st.header("Documentation Helper Bot")

prompt = st.text_input("prompt", placeholder="Ask me anything about langchain!")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "bot_response_history" not in st.session_state:
    st.session_state["bot_response_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


if prompt:
    ## with notation allows for the spinner to dissapear after code inside is finished
    with st.spinner("Generating Response"):
        generated_repsonse = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        ## we use set to remove any duplicates
        sources = set(
            [doc.metadata["source"] for doc in generated_repsonse["source_documents"]]
        )

        formatted_response = (
            f'{generated_repsonse["answer"]} \n \n ' f"{create_sources_string(sources)}"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["bot_response_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_repsonse["answer"]))
        st.session_state["prompt"] = ""

if st.session_state["bot_response_history"]:
    for prompt, response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["bot_response_history"],
    ):
        message(prompt, is_user=True)
        message(response)
