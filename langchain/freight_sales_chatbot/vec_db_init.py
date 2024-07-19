import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_BASE"] = "https://aigptx.top/v1"
os.environ["OPENAI_API_KEY"] = "sk-hFVEtXdOCaB6DF955908T3BlbKFJ17ce4e69d6ed4fc58540"
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
