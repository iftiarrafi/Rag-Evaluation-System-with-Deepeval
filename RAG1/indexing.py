import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from RAG1.file_reader import extract_paragraphs

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-embedding-2-preview",
        output_dimensionality=768
)

vector_store = InMemoryVectorStore(embedding=embedding)

path = r"pdfs\\Penguins_ACL.pdf"
chunks = extract_paragraphs(pdf_path=path)
vector_store.add_texts(chunks)
