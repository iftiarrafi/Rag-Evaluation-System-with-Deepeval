import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer:str = Field(description="the answer that is needed for the user question")
    context:str = Field(description="this is the context that the agent used to reply to the user question")