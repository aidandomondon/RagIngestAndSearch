from embeddors.Embeddor import Embeddor
from vectordbs.VectorDB import VectorDB
from prompts.Prompt import Prompt
from llms.LLM import LLM
from Searcher import search_all
from time import perf_counter_ns

def measure_query(
    queries: list[str], embeddor: Embeddor, vector_db: VectorDB, prompt: Prompt, llm: LLM
) -> dict:
    """
    `queries`: list of questions to ask the LLM \\
    `vector_db`: vector database where context embeddings are stored \\
    `prompt`: type of prompt to compile the question and context into before sending to the LLM. \\
    `llm`: LLM to ask
    """
    start = perf_counter_ns()
    responses = search_all(queries, embeddor, vector_db, prompt, llm)
    end = perf_counter_ns()
    query_time = int((end - start) / 1e6)
    
    return {
        "query_time": query_time,
        "responses": responses
    }