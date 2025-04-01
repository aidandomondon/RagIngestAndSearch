from llms.LLM import LLM
from ollama import chat

class OllamaLLM(LLM):
    """
    An LLM run on Ollama.
    """

    def __init__(self, model: str):
        self.model = model # the name of the LLM to be used, as it is referred to on Ollama

    def generate_rag_response(self, prompt: str) -> str:
        # Generate response using Ollama
        response = chat(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]