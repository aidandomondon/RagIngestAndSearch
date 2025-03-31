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

# Text preparation steps, 
# chunk size & chunk overlap size, 
# the vector database,
# the prompt template,
# and the LLM are all held constant
text_preparer = TextPreparer(
    remove_whitespace=False,
    remove_punctuation=False,
    remove_stopwords=False,
    lemmatize=False
)
chunker = BasicChunker()
vector_db = RedisVectorDB(-1)
prompt_template = BasicPrompt()
llm = OllamaLLM('llama3.2')

# The embedding model is varied
models = {
    "all-minilm": MiniLMEmbeddor(),
    "all-minilm:33m": MiniLM33Embeddor(),
    "nomic-embed-text": NomicEmbeddor()
}

test_queries = [
    "List three differences between Redis and MongoDB.",
    "Name one way in which a linked-list is better than a continguous array.",
    "What is the name of the library that allows us to interact with a MongoDB in Python?",
    "What is the name of a query language for Neo4J?",
    "What is the difference between a B tree and a B+ tree?"
]

results = {}
for model_name, model in tqdm(sample(sorted(models.items()), k=len(models)), unit='model'):
    print(f'Testing the embedding model {model_name}...')
    # Test ingestion speed and memory usage
    stats = {}
    print('Testing ingestion speed and memory usage...')
    stats.update(
        measure_ingest(
            text_preparer=text_preparer, chunker=chunker, vector_db=vector_db,
            embeddor = model
        )
    )
    print('Done testing ingestion.')
    print('Testing query speed...')
    stats.update(
        measure_query(
            queries=test_queries, vector_db=vector_db, prompt=prompt_template, llm=llm,
            embeddor = model
        )
    )
    print('Done testing query speed.')
    results[model_name] = stats
    print(f'Done testing {model_name}.')

# Print results
print("Model\tIngestion Time\tQuery Time\tMemory Usage")
for model_name, result in results.items(): 
    print(f"{model_name}\t{result['ingestion_time']}\t{result['query_time']}\t{result['memory_usage']}")