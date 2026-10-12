import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))

from RAG1.indexing import vector_store
from langchain.tools import tool

@tool
def collecting_info(query:str , k=2) -> str:
    """This tool extracts the information from vectorDB by similarity search by user query

    Args:
        query (str): 

    Returns:
        str:
    """
    
    global last_retrieved_context
    
    docs = vector_store.similarity_search(query=query , k=k)
    
    last_retrieved_context = [doc.page_content for doc in docs]
    
    return "\n\n".join(last_retrieved_context)