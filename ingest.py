from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader,
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constant import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # create embeddings here
    embeddings = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
    
    # create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, clinet_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 


if __name__ == "__main__":
    main()