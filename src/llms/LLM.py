from abc import ABC, abstractmethod

class LLM(ABC):
    
    @abstractmethod
    def generate_rag_response(self, prompt: str) -> str:
        """
        Returns the LLM's repsonse to the given prompt.
        """
        ...