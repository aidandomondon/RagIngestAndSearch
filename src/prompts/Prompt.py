from abc import ABC, abstractmethod

class Prompt(ABC):
    
    @abstractmethod
    def compile_prompt(self, query: str, context_results: list) -> str:
        """
        Generate a prompt for an LLM, instructing it to consult
        `context_results` to answer `query`.
        """
        ...