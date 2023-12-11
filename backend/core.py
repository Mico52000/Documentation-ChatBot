import os

from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
import typing

from consts import INDEX_NAME


load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def run_llm(query: str) -> any:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = os.getenv("HF_API_KEY")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    gpt4all_path = os.getenv("GPT4ALL_PATH")
    llm_path = os.path.join(gpt4all_path, "nous-hermes-llama2-13b.Q4_0.gguf")
    llm = GPT4All(
        temp=0,
        model=llm_path,
        verbose=True,
        streaming=True,
        max_tokens=4000,

    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm("what are the type of chains in langchain"))
