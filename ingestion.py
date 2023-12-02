import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
import pinecone
load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
INDEX_NAME = "langchain-docs-index"


def ingest_docs():
    doc_loader = ReadTheDocsLoader(path='langchain-docs/api.python.langchain.com/en/latest')
    raw_docs = doc_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_docs)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = os.getenv("HF_API_KEY")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print(f'${len(documents)} moved to vector store on pinecone')





if __name__ == '__main__':
    print(ingest_docs())
