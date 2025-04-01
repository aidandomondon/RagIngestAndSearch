from embeddors.Embeddor import Embeddor
from vectordbs.VectorDB import VectorDB
from prompts.Prompt import Prompt
from llms.LLM import LLM
from tqdm import tqdm

def search_all(
    queries: list[str], embeddor: Embeddor, vector_db: VectorDB, prompt: Prompt, llm: LLM
) -> dict:
    """
    `queries`: list of questions to ask the LLM \\
    `vector_db`: vector database where context embeddings are stored \\
    `prompt`: type of prompt to compile the question and context into before sending to the LLM. \\
    `llm`: LLM to ask
    """
    responses = {}
    for query in tqdm(queries, unit='query'):
        # Embed query text
        query_embedding = embeddor.get_embedding(query)

        # Search for vectors similar to the query vector.
        context_results = vector_db.search_embeddings(query_embedding)

        # Combine these similar vectors together with the original query text
        # for a compiled prompt.
        compiled_prompt = prompt.compile_prompt(query, context_results)
        
        # Ask the LLM the compiled prompt.
        responses[query] = llm.generate_rag_response(compiled_prompt)

    return responses