import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))
from RAG1.tools import collecting_info
import RAG1.tools as tl
from RAG1.prompts import system_agent_prompt
from RAG1.objects import Answer
from langchain.messages import HumanMessage
from langchain.agents.structured_output import ToolStrategy

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_groq import ChatGroq

tools = [collecting_info]

llm = ChatGroq(model="openai/gpt-oss-120b" ,
            api_key=os.getenv("GROQ_API_KEY") , 
            temperature=0)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_agent_prompt,
    response_format=ToolStrategy(Answer),
)

def ask(question : str) -> dict :
    """ Answer the user's question about the pdf"""
    tl.last_retrieved_context = []
    
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    
    structured_response = result["structured_response"]
    
    return {
        "answer" : structured_response.answer,
        "context" : structured_response.context,
    }
    
if __name__ == "__main__":
    print(ask("what are the diets of the penguins?"))


