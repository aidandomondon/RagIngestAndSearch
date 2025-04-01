from prompts.Prompt import Prompt
from textwrap import dedent

class BasicPrompt(Prompt):

    def compile_prompt(self, query: str, context_results: list) -> str:

        # Prepare context string
        context_str = "\n".join(
            [
                f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
                f"with similarity {float(result.get('similarity', 0)):.2f}"
                for result in context_results
            ]
        )

        # Construct prompt with context
        prompt = dedent(f"""\
                        You are a helpful AI assistant. 
                        Use the following context to answer the query as accurately as possible. If the context is 
                        not relevant to the query, say 'I don't know'.

                        Context:
                        {context_str}

                        Query: {query}

                        Answer:""")
        
        return prompt