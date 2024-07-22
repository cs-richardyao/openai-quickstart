import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


with open("container_agent.txt", encoding='utf8', errors="ignore") as f:
    container_agent = f.read()
text_splitter = CharacterTextSplitter(
    separator=r'\d+',
    chunk_size=150,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True,
)
docs = text_splitter.create_documents([container_agent])
db = FAISS.from_documents(docs, OpenAIEmbeddings())
db.save_local("container_agent")
