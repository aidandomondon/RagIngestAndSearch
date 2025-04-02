from prompts.BasicPrompt import BasicPrompt
from textwrap import dedent

class OutsideKnowledgePrompt(BasicPrompt):


    def compile_prompt(self, query: str, context_results: list) -> str:

        # Prepare context string
        context_str = super().compile_context_str(context_results)

        # Construct prompt with context
        prompt = dedent(f"""\
                        You are a helpful AI assistant. 
                        Use the following context to answer the query as accurately as possible. If the context is 
                        not relevant to the query, use other outside knowledge you may have to answer the question.

                        Context:
                        {context_str}

                        Query: {query}

                        Answer:""")
        
        return prompt