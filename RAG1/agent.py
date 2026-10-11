import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))
from RAG1.tools import collecting_info
import RAG1.tools as tl
from RAG1.prompts import system_agent_prompt 
from RAG1.objects import Answer
from langchain.messages import HumanMessage , SystemMessage
from langchain.agents.structured_output import ToolStrategy

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_groq import ChatGroq


llm = ChatGroq(model="openai/gpt-oss-120b" ,
            api_key=os.getenv("GROQ_API_KEY") , 
            temperature=0
)

structured_llm = llm.with_structured_output(Answer ,method="json_mode")

def ask(question : str) -> dict :
    """ Answer the user's question about the pdf"""
    
    retrieved_context = collecting_info.invoke(
        {"query": question}
    )
    
    system_message = SystemMessage(
        content=(
            f"{system_agent_prompt}\n\n"
            "Return only a valid JSON object with these two string "
            "fields: answer and context.\n"
            "The answer must be based only on the retrieved context.\n"
            "The context field must contain the retrieved PDF text."
        )
    )
    
    human_message = HumanMessage(
        content=(
            f"Question:\n{question}\n\n"
            f"Retrieved PDF context:\n{retrieved_context}"
        )
    )
    
    response = structured_llm.invoke(
        [system_message, human_message]
    )
    
    # print(type(response))
    
    return {
        "answer": response.answer,
        "context": response.context,
    }
    
# if __name__ == "__main__":
#     ask("what are the diets of the penguins?")


