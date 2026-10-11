import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str = Field(
        description="The answer to the user's question based only on the retrieved PDF context."
    )

    context: str = Field(
        description="The retrieved PDF context used to answer the question."
    )