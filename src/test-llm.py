from tqdm import tqdm
from textpreparers.TextPreparer import TextPreparer
from chunkers.BasicChunker import BasicChunker
from vectordbs.RedisVectorDB import RedisVectorDB
from prompts.BasicPrompt import BasicPrompt
from llms.OllamaLLM import OllamaLLM
from embeddors.MiniLMEmbeddor import MiniLMEmbeddor
from embeddors.MiniLM33Embeddor import MiniLM33Embeddor
from embeddors.NomicEmbeddor import NomicEmbeddor
from random import sample
from measured_ingest import measure_ingest
from measured_query import measure_query
from os.path import join
from json import dumps

# Constant text preparation, chunking, vector database, and prompt template
text_preparer = TextPreparer(
    remove_whitespace=False,
    remove_punctuation=False,
    remove_stopwords=False,
    lemmatize=False
)
chunker = BasicChunker()
vector_db = RedisVectorDB(-1)
prompt_template = BasicPrompt()

# Use a constant embedding model
embedding_model = MiniLMEmbeddor()

# Varying the LLMs
llms = {
    "llama3.2:latest": OllamaLLM("llama3.2:latest"),
    "deepseek-r1:8b": OllamaLLM("deepseek-r1:8b")
}

test_queries = [
    "List three differences between Redis and MongoDB.",
    "Name one way in which a linked-list is better than a continguous array.",
    "What is the name of the library that allows us to interact with a MongoDB in Python?",
    "What is the name of a query language for Neo4J?",
    "What is the difference between a B tree and a B+ tree?"
]

results = {}
for llm_name, llm_instance in tqdm(sample(sorted(llms.items()), k=len(llms)), unit='model'):
    print(f'Testing the LLM {llm_name}...')
    stats = {}
    
    print('Testing ingestion speed and memory usage...')
    stats.update(
        measure_ingest(
            text_preparer=text_preparer, chunker=chunker, vector_db=vector_db,
            embeddor=embedding_model  # constant embedding model used for ingestion
        )
    )
    print('Done testing ingestion.')
    
    print('Testing query speed...')
    stats.update(
        measure_query(
            queries=test_queries, vector_db=vector_db, prompt=prompt_template, llm=llm_instance,
            embeddor=embedding_model  # constant embedding model used for querying
        )
    )
    print('Done testing query speed.')
    
    results[llm_name] = stats
    print(f'Done testing {llm_name}.')

# Print and save results
print("LLM\tIngestion Time\tQuery Time\tMemory Usage")
for llm_name, result in results.items():
    with open(join(".", "llm_responses", f"test-llm-{llm_name}.json"), 'w') as file:
        file.write(dumps(result['responses']))
    print(f"{llm_name}\t{result['ingestion_time']}\t{result['query_time']}\t{result['memory_usage']}")
