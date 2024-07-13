"""
Document retriever

"""
import argparse
import os
import sys
from uuid import uuid4

from dotenv import load_dotenv
from fake_useragent import UserAgent
from langchain.chains import RetrievalQA
from langchain.retrievers.contextual_compression import \
    ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

# load .env
load_dotenv()

# API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Langchain LangSmith
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Document URL (Wikipedia)
DOCUMENT_URL = "https://ja.wikipedia.org/wiki/%E5%8C%97%E9%99%B8%E6%96%B0%E5%B9%B9%E7%B7%9A"

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('query', help='Query string')
parser.add_argument('-q', '--exec_query', action='store_true', help='execute query')
parser.add_argument('-v', '--retriever', action='store_true', help='Retrieve with normal vector search')
parser.add_argument('-r', '--rerank', action='store_true', help='Retrieve with vector search + reranker')

# retriever settings
TOP_K_VECTOR = 10
TOP_K_RERANK = 3

def load_and_split_document(url: str) -> list[Document]:
    """Load and split document

    Args:
        url (str): Document URL

    Returns:
        list[Document]: splitted documents
    """

    # Read the Wep documents from 'url'
    raw_documents = WebBaseLoader(url, header_template={
      'User-Agent': UserAgent().chrome,
        }).load()

    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    # Split the documents
    documents = text_splitter.split_documents(raw_documents)

    # for TEST
    print("Original document: ", len(documents), " docs")

    return documents


def vector_retriever():
    """Create base vector retriever

    Returns:
        Vector Retriever
    """

    # load and split document
    documents = load_and_split_document(DOCUMENT_URL)

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents, embeddings)

    # base retriever (vector retriever)
    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": TOP_K_VECTOR},
    )

    return vector_retriever


def retriever_with_rerank():
    """
    Retrieve documents using Multivactor retriever with Rerank
    """

    # base retriever
    base_retriever = vector_retriever()

    # Reranker
    cohere_reranker = CohereRerank(
        top_n=TOP_K_RERANK,
        cohere_api_key=COHERE_API_KEY)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor = cohere_reranker,
        base_retriever = base_retriever,
    )

    return compression_retriever


def rag(retriever, query: str):

    # LLM model
    llm_model = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo")

    # RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever)

    # invoke
    result = chain.invoke({"query": query})

    return result.get('result')


# main
def main():

    args = parser.parse_args()

    # query string
    query = args.query

    # retriever
    retriever = None
    if args.retriever:
        # vector retriever
        retriever = vector_retriever()
    elif args.rerank:
        # vector retriever + reranker
        retriever = retriever_with_rerank()
    
    if not retriever:
        print("Error: no retriever set.")
        sys.exit(1)

    if args.exec_query:
        # RAG query
        result = rag(retriever, query)
        print(result)
    
    else:
        # Retrieve only
        docs = retriever.get_relevant_documents(query)

        for doc in docs:
            title = doc.metadata['title']
            content = doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
            print(f"Title: {title}\nContent: {content}\n")


if __name__ == '__main__':
    main()
