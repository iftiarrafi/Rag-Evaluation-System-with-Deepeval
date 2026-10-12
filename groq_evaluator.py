from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

class GroqEvaluator(DeepEvalBaseLLM):

    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        self.llm = ChatGroq(
            model=model_name,
            temperature=0
        )

    def load_model(self):
        return self.llm

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content